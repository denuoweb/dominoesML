from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import os

def load_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.flatten())
                # Correctly extracting the dot count
                label_part = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
                try:
                    dot_count = int(label_part)
                    labels.append(dot_count)
                except ValueError:
                    print(f"Error converting {label_part} to int for file: {filename}")
                    continue  # Skip this file if the label cannot be converted to an integer
    return np.array(images), np.array(labels)

# Load training and validation data
X_train, y_train = load_images('basePhotos/train')
X_val, y_val = load_images('basePhotos/validate')

# Combine training and validation sets
X_combined = np.concatenate((X_train, X_val), axis=0)
y_combined = np.concatenate((y_train, y_val), axis=0)

# Since the dataset is small, use StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=min(2, len(np.unique(y_combined))), shuffle=True, random_state=42)

# Initialize the DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, X_combined, y_combined, cv=skf)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

# Training the model on the combined dataset
model.fit(X_combined, y_combined)

# Optionally, evaluate on an external test set if available
# X_test, y_test = load_images('path_to_test_set')
# predictions = model.predict(X_test)
# print("Test set accuracy:", accuracy_score(y_test, predictions))
