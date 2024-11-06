import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from scipy import stats
from keras.models import load_model
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
# from scipy.stats import mcnemar, chi2_contingency, friedmanchisquare, kruskal
from scipy.stats import chi2_contingency, friedmanchisquare, kruskal
from statsmodels.stats.contingency_tables import mcnemar 
import json
import tensorflow_hub as hub
import joblib
# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    directory="Dataset A/test",
    target_size=(224, 224),
    batch_size=8,
    shuffle=False,
    class_mode='categorical'
)
test_gen2 = test_datagen.flow_from_directory(
    directory="Dataset A/test",
    target_size=(256, 256),
    batch_size=8,
    shuffle=False,
    class_mode='categorical'
)
custom_object = {'KerasLayer': hub.KerasLayer}
model1 = load_model('models/DEiT/model_1.h5', custom_objects=custom_object)
custom_object = {'KerasLayer': hub.KerasLayer}
model2 = load_model('models/mobilevit_v2/model_2.h5', custom_objects=custom_object)
custom_object = {'KerasLayer': hub.KerasLayer}
model3 = load_model('models/swin/saved model/model_1.h5', custom_objects=custom_object)
print(1)
# Function to get predictions for each model
def get_predictions(model, test_gen):
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes
true_classes = test_gen.classes

# Get predictions for each model
# predictions1 = get_predictions(model1, test_gen)
# predictions2 = get_predictions(model2, test_gen2)
# predictions3 = get_predictions(model3, test_gen)
predictions1=joblib.load("p1.joblib")
predictions2=joblib.load("p2.joblib")
predictions3=joblib.load("p3.joblib")

# predictions4 = get_predictions(model4, test_gen)
# Perform ANOVA
# Prepare boolean arrays indicating correct predictions
correct_predictions1 = predictions1 == true_classes
correct_predictions2 = predictions2 == true_classes
correct_predictions3 = predictions3 == true_classes
f_value, p_value_anova = stats.f_oneway(correct_predictions1, correct_predictions2, correct_predictions3)
joblib.dump(predictions1,"p1.joblib")
joblib.dump(predictions2,"p2.joblib")
joblib.dump(predictions3,"p3.joblib")

# Perform Tukey's HSD if ANOVA is significant
if p_value_anova < 0.05:
    # Combine predictions into a single array
    all_correct_predictions = np.concatenate([correct_predictions1, correct_predictions2, correct_predictions3])

    # Create an array of labels corresponding to each group of predictions
    all_labels = np.concatenate([
        np.full(correct_predictions1.shape, "Model1"),
        np.full(correct_predictions2.shape, "Model2"),
        np.full(correct_predictions3.shape, "Model3")
    ])

    mc = MultiComparison(all_correct_predictions, all_labels)
    tukey_result = mc.tukeyhsd()
else:
    tukey_result = "ANOVA not significant, Tukey's HSD not performed"

# Get true classes
true_classes = test_gen.classes

# Perform ANOVA
# f_value, p_value_anova = stats.f_oneway(predictions1 == true_classes, predictions2 == true_classes, predictions3 == true_classes, predictions4 == true_classes)
# f_value, p_value_anova = stats.f_oneway(predictions1 == true_classes, predictions2 == true_classes, predictions3 == true_classes)

# # Perform Tukey's HSD if ANOVA is significant
# if p_value_anova < 0.05:
#     # all_predictions = np.concatenate([predictions1, predictions2, predictions3, predictions4])
#     all_predictions = np.concatenate([predictions1, predictions2, predictions3])

#     # all_labels = np.concatenate([np.full(predictions1.shape, "Model1"), np.full(predictions2.shape, "Model2"), np.full(predictions3.shape, "Model3"), np.full(predictions4.shape, "Model4")])
#     all_labels = np.concatenate([np.full(predictions1.shape, "Model1"), np.full(predictions2.shape, "Model2"), np.full(predictions3.shape, "Model3")])

#     mc = MultiComparison(all_predictions == true_classes, all_labels)
#     tukey_result = mc.tukeyhsd()
# else:
#     tukey_result = "ANOVA not significant, Tukey's HSD not performed"

# Prepare comparisons for T-test and Mann-Whitney U Test
# models = ['Model1', 'Model2', 'Model3', 'Model4']
models = ['Model1', 'Model2', 'Model3']

# all_predictions = [predictions1, predictions2, predictions3, predictions4]
all_predictions = [predictions1, predictions2, predictions3]

# Independent T-tests
t_test_results = {}
for i in range(len(models)):
    for j in range(i+1, len(models)):
        t_stat, p_val = stats.ttest_ind(all_predictions[i] == true_classes, all_predictions[j] == true_classes)
        t_test_results[f'{models[i]} vs {models[j]}'] = p_val

# Mann-Whitney U Tests
mann_whitney_results = {}
for i in range(len(models)):
    for j in range(i+1, len(models)):
        u_stat, p_val = stats.mannwhitneyu(all_predictions[i] == true_classes, all_predictions[j] == true_classes)
        mann_whitney_results[f'{models[i]} vs {models[j]}'] = p_val

# McNemar's Test
# mcnemar_results = {}
# for i in range(len(models)):
#     for j in range(i+1, len(models)):
#         contingency_table = pd.crosstab(all_predictions[i] == true_classes, all_predictions[j] == true_classes)
#         stat, p_val = mcnemar(contingency_table)
#         mcnemar_results[f'{models[i]} vs {models[j]}'] = p_val
# McNemar's Test
mcnemar_results = {}
for i in range(len(models)):
    for j in range(i+1, len(models)):
        contingency_table = pd.crosstab(all_predictions[i] == true_classes, all_predictions[j] == true_classes)
        mcnemar_test = mcnemar(contingency_table)
        mcnemar_results[f'{models[i]} vs {models[j]}'] = mcnemar_test.pvalue

# Chi-Square Test of Independence
# chi2_stat, chi2_p_val, _, _ = chi2_contingency(pd.crosstab(all_predictions.flatten(), true_classes))
# Chi-Square Test of Independence
combined_predictions = np.concatenate(all_predictions)
# chi2_stat, chi2_p_val, _, _ = chi2_contingency(pd.crosstab(combined_predictions, true_classes))

# Friedman Test
# friedman_stat, friedman_p_val = friedmanchisquare(predictions1 == true_classes, predictions2 == true_classes, predictions3 == true_classes, predictions4 == true_classes)
friedman_stat, friedman_p_val = friedmanchisquare(predictions1 == true_classes, predictions2 == true_classes, predictions3 == true_classes)

# Kruskal-Wallis H Test
# kruskal_stat, kruskal_p_val = kruskal(predictions1 == true_classes, predictions2 == true_classes, predictions3 == true_classes, predictions4 == true_classes)
kruskal_stat, kruskal_p_val = kruskal(predictions1 == true_classes, predictions2 == true_classes, predictions3 == true_classes)

# Save results
results = {
    "ANOVA": p_value_anova,
    "Tukey HSD": str(tukey_result),
    "T-tests": t_test_results,
    "Mann-Whitney U": mann_whitney_results,
    "McNemar's Test": mcnemar_results,
    # "Chi-Square Test": chi2_p_val,
    "Friedman Test": friedman_p_val,
    "Kruskal-Wallis H Test": kruskal_p_val
}

# Save to a file
with open('model_comparison_results.json', 'w') as f:
    json.dump(results, f)

print("Results saved successfully.")
import numpy as np
import joblib

# Function to calculate accuracy
def calculate_accuracy(predictions, true_classes):
    correct_predictions = predictions == true_classes
    accuracy = np.sum(correct_predictions) / len(true_classes)
    return accuracy

# Load predictions and true classes
predictions1 = joblib.load("p1.joblib")
predictions2 = joblib.load("p2.joblib")
predictions3 = joblib.load("p3.joblib")
# Assuming true_classes is a numpy array of true class labels

# Calculate accuracy for each model
accuracy_model1 = calculate_accuracy(predictions1, true_classes)
accuracy_model2 = calculate_accuracy(predictions2, true_classes)
accuracy_model3 = calculate_accuracy(predictions3, true_classes)

# Print the accuracies
print(f"Accuracy of Model 1: {accuracy_model1:.4f}")
print(f"Accuracy of Model 2: {accuracy_model2:.4f}")
print(f"Accuracy of Model 3: {accuracy_model3:.4f}")
