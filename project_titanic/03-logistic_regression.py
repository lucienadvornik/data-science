import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#### MODEL TRAINING ####
from sklearn.linear_model import LogisticRegression
# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)


#### MODEL EVALUATION ####
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print the accuracy
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  # Print the confusion matrix


# Make predictions on the test set
y_pred = model.predict(X_test)

# Precision, Recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))


## Results
# Accuracy: 0.79
# Confusion Matrix:
# [[89 16] = 89 TP, 16 FP
# [20 54]] = 20 FN, 54 TN
# Precision: 0.7714285714285715 (correctly predicted positive examples)
# Recall: 0.7297297297297297 (how well the model can find all true positive examples)
# F1-Score: 0.75 (A harmonic mean between precision and recall that provides a balanced view of model precision and sensitivity)
