import os
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_DEBUG'] = 'INFO'
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, DistributedSampler
import timm
from tqdm import tqdm
import numpy as np

NUM_CLASSES = 5
NUM_EPOCHS = 25
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

def main_worker(gpu, ngpus_per_node):
    setup(gpu, ngpus_per_node)
    torch.cuda.set_device(gpu)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root='classification_aunet_clahe_seg_data', transform=transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    criterion = nn.CrossEntropyLoss()
    torch.cuda.synchronize()
    for fold in range(K_FOLDS):
        torch.cuda.synchronize()
        # Split indices for k-fold cross-validation
        print("next fold")
        fold_size = dataset_size // K_FOLDS
        start, end = fold * fold_size, (fold + 1) * fold_size
        test_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]

        train_sampler = DistributedSampler(Subset(dataset, train_indices), num_replicas=ngpus_per_node, rank=gpu)
        test_sampler = DistributedSampler(Subset(dataset, test_indices), num_replicas=ngpus_per_node, rank=gpu, shuffle=False)

        train_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, sampler=test_sampler)

        model = timm.create_model('mobilevit_xs', pretrained=True, num_classes=NUM_CLASSES).cuda(gpu)
        model = DDP(model, device_ids=[gpu])
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Early stopping setup
        best_loss = float('inf')
        patience_counter = 0
        torch.cuda.synchronize()
        for epoch in range(NUM_EPOCHS):
            torch.cuda.synchronize()
            model.train()
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            correct_predictions = 0
            total_train_samples = 0

            with tqdm(total=len(train_loader), desc=f"Fold {fold+1}/{K_FOLDS}, Epoch {epoch+1}/{NUM_EPOCHS}", disable=gpu != 0) as pbar:
                torch.cuda.synchronize()
                for images, labels in train_loader:
                    images, labels = images.cuda(gpu), labels.cuda(gpu)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs.data, 1)
                    total_train_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()
                    epoch_loss += loss.item() * images.size(0)  # Update to accumulate loss correctly

                    pbar.update(1)
                torch.cuda.synchronize()
            # Calculate average loss and accuracy over the epoch
            avg_train_loss = epoch_loss / total_train_samples
            train_accuracy = 100.0 * correct_predictions / total_train_samples

            # Validation phase
            if gpu == 0:  # Perform validation only on the main GPU
                val_loss, val_accuracy = validate(model, test_loader, criterion, gpu)
                print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

                # Early stopping logic based on validation loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), f'fold_{fold+1}_best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"Early stopping triggered on fold {fold+1} after {epoch+1} epochs.")
                        break  # Exit the epoch loop for early stopping
    cleanup()

if __name__ == "__main__":
    ngpus_per_node = torch.cuda.device_count()
    print(f"Using {ngpus_per_node} GPUs.")
    mp.spawn(main_worker, args=(ngpus_per_node,), nprocs=ngpus_per_node, join=True)
