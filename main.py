import os
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

# Use inline plots if running in Jupyter
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

# ---------------------------
# File Paths
# ---------------------------

# Define all relative paths
serum_csv_path = "data/pure/serum.csv"
oil_csv_path = "data/pure/oil.csv"
mixture_dir = "data/mixtures/"
mixture_files = [
    "10_1_10805.csv", "10_2_10805.csv", "10_3_10805.csv",
    "15_1_10805.csv", "15_2_10805.csv", "15_3_10805.csv",
    "30_1_10805.csv", "30_2_10805.csv", "30_3_10805.csv",
    "35_1_10805.csv", "35_2_10805.csv", "35_3_10805.csv"
]
mixture_paths = [os.path.join(mixture_dir, file) for file in mixture_files]

# ---------------------------
# Load Pure Serum and Oil Data
# ---------------------------

if not os.path.exists(serum_csv_path):
    raise FileNotFoundError(f"Serum file not found at {serum_csv_path}")
if not os.path.exists(oil_csv_path):
    raise FileNotFoundError(f"Oil file not found at {oil_csv_path}")

serum_raw = pd.read_csv(serum_csv_path)
oil_raw = pd.read_csv(oil_csv_path)

# ---------------------------
# Augmentation Function
# ---------------------------

def augment_data(df, num_samples, noise_level=0.01):
    augmented_data = df.copy()
    for _ in range(num_samples):
        noisy = df.copy()
        noisy['RamanShift'] += np.random.normal(0, noise_level * df['RamanShift'].std(), size=df.shape[0])
        noisy['Intensity'] += np.random.normal(0, noise_level * df['Intensity'].std(), size=df.shape[0])
        augmented_data = pd.concat([augmented_data, noisy], axis=0)
    return augmented_data.sample(frac=1).reset_index(drop=True)

# ---------------------------
# Augment Pure Samples
# ---------------------------

serum_data = augment_data(serum_raw, num_samples=10)
serum_data['Label'] = 'Serum'

oil_data = augment_data(oil_raw, num_samples=10)
oil_data['Label'] = 'Oil'

# ---------------------------
# Plot Augmentation Comparison
# ---------------------------

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(serum_raw['RamanShift'], serum_raw['Intensity'], label='Original Serum')
plt.plot(oil_raw['RamanShift'], oil_raw['Intensity'], label='Original Oil')
plt.title('Before Augmentation')
plt.xlabel('Raman Shift')
plt.ylabel('Intensity')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(serum_data['RamanShift'], serum_data['Intensity'], label='Augmented Serum', alpha=0.6)
plt.plot(oil_data['RamanShift'], oil_data['Intensity'], label='Augmented Oil', alpha=0.6)
plt.title('After Augmentation')
plt.xlabel('Raman Shift')
plt.ylabel('Intensity')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# Load & Augment Mixture Samples
# ---------------------------

augmented_mixtures = pd.DataFrame()
for file_path in mixture_paths:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mixture file not found: {file_path}")
    mixture_df = pd.read_csv(file_path)
    augmented = augment_data(mixture_df, num_samples=10)
    augmented['Label'] = 'Mixture'
    augmented_mixtures = pd.concat([augmented_mixtures, augmented], axis=0)

# ---------------------------
# Combine & Clean Data
# ---------------------------

combined_data = pd.concat([serum_data, oil_data, augmented_mixtures], axis=0).reset_index(drop=True)
combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna()

# ---------------------------
# Median Filter
# ---------------------------

combined_data['Intensity'] = median_filter(combined_data['Intensity'], size=5)
print("Smoothing Complete.")

# ---------------------------
# Normalize and Reduce Dimensions
# ---------------------------

scaler = RobustScaler()
normalized = scaler.fit_transform(combined_data[['RamanShift', 'Intensity']])
selected_data = pd.DataFrame(normalized, columns=['RamanShift', 'Intensity'])
selected_data['Label'] = combined_data['Label']

pca = PCA(n_components=2)
pca_result = pca.fit_transform(selected_data[['RamanShift', 'Intensity']])
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['Label'] = selected_data['Label']

# ---------------------------
# Split Data
# ---------------------------

X = pca_df[['PC1', 'PC2']]
y = pca_df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------
# Train and Evaluate Models
# ---------------------------

models = {
    'RandomForest': RandomForestClassifier(random_state=0),
    'SVM': SVC(kernel='linear', random_state=0),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'LogisticRegression': LogisticRegression(random_state=0),
    'DecisionTree': DecisionTreeClassifier(random_state=0),
    'GradientBoosting': GradientBoostingClassifier(random_state=0),
    'NaiveBayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(random_state=0),
    'ExtraTrees': ExtraTreesClassifier(random_state=0)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    prec = precision_score(y_test, y_pred, average='weighted') * 100
    results[name] = {
        'Accuracy (%)': f"{acc:.2f}",
        'F1 Score (%)': f"{f1:.2f}",
        'Precision (%)': f"{prec:.2f}"
    }
    print(f"Model: {name}, Accuracy: {acc:.2f}%, F1 Score: {f1:.2f}%, Precision: {prec:.2f}%")

# ---------------------------
# Best Model Evaluation
# ---------------------------

best_model_name = max(results, key=lambda x: float(results[x]['F1 Score (%)']))
print(f"\nBest Model: {best_model_name}")
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title("Confusion Matrix for Best Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Save Results
results_df = pd.DataFrame(results).T
results_df.to_csv('model_evaluation_results.csv', index=True)
print("Saved: model_evaluation_results.csv")

# Accuracy Bar Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=list(results.keys()), y=[float(results[m]['Accuracy (%)']) for m in results], palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.ylim(0, 110)
plt.tight_layout()
plt.show()
