import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, DistributedSampler
import timm
from tqdm import tqdm
import numpy as np

# Set environment variables before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'   
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_DEBUG'] = 'INFO'

NUM_CLASSES = 4
NUM_EPOCHS = 2
K_FOLDS = 10
PATIENCE = 3  # Number of epochs to wait before early stopping if no improvement in validation loss

def validate(model, loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct_predictions / total
    return avg_loss, accuracy

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root='path', transform=transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    criterion = nn.CrossEntropyLoss()

    try:
        for fold in range(K_FOLDS):
            # Split indices for k-fold cross-validation
            print(f"Starting Fold {fold+1}/{K_FOLDS}")
            fold_size = dataset_size // K_FOLDS
            start, end = fold * fold_size, (fold + 1) * fold_size
            test_indices = indices[start:end]
            train_indices = indices[:start] + indices[end:]

            train_subset = Subset(dataset, train_indices)
            test_subset = Subset(dataset, test_indices)

            train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
            test_sampler = DistributedSampler(test_subset, num_replicas=world_size, rank=rank, shuffle=False)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
            test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, sampler=test_sampler)

            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES).cuda(rank)
            model = DDP(model, device_ids=[rank])
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Early stopping setup
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(NUM_EPOCHS):
                model.train()
                train_sampler.set_epoch(epoch)
                epoch_loss = 0.0
                correct_predictions = 0
                total_train_samples = 0

                with tqdm(total=len(train_loader), desc=f"Fold {fold+1}/{K_FOLDS}, Epoch {epoch+1}/{NUM_EPOCHS}", disable=rank != 0) as pbar:
                    for images, labels in train_loader:
                        images, labels = images.cuda(rank, non_blocking=True), labels.cuda(rank, non_blocking=True)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        _, predicted = torch.max(outputs.data, 1)
                        total_train_samples += labels.size(0)
                        correct_predictions += (predicted == labels).sum().item()
                        epoch_loss += loss.item() * images.size(0)  # Accumulate loss correctly

                        pbar.update(1)

                # Calculate average loss and accuracy over the epoch
                avg_train_loss = epoch_loss / total_train_samples
                train_accuracy = 100.0 * correct_predictions / total_train_samples

                # Validation phase
                if rank == 0:  # Perform validation only on the main GPU
                    val_loss, val_accuracy = validate(model, test_loader, criterion, rank)
                    print(f'Fold {fold+1}, Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

                    # Early stopping logic based on validation loss
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(model.module.state_dict(), f'fold_{fold+1}_best_model.pth')
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= PATIENCE:
                            print(f"Early stopping triggered on fold {fold+1} after {epoch+1} epochs.")
                            break  # Exit the epoch loop for early stopping

                # **Add synchronization barrier after validation**
                dist.barrier()

            # Save the model after each fold (if rank 0)
            if rank == 0:
                torch.save(model.module.state_dict(), f'fold_{fold+1}_vit_model.pth')

            # **Add synchronization barrier after saving the model and before breaking**
            dist.barrier()

            # **Break the fold loop (as per your intention)**
            break

        # **Final synchronization before cleanup**
        dist.barrier()

    except Exception as e:
        print(f"An error occurred on rank {rank}: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs.")
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
