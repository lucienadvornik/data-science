#### MACHINE LEARNING MODEL ####
## with export for Kaggle competition ##
import pandas as pd 
import numpy as numpy
#import os

# Import the RandomForestClassifier from scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Define target variable 'y' as the 'Survived' column
y = train["Survived"]

# Select relevant features for the model
features = ["Pclass", "Sex", "SibSp", "Parch"]

# Convert categorical variables into dummy variables
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

# Initialize RandomForest model with 100 trees and max depth of 5
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Train the model on the training data
model.fit(X, y)

# Make predictions on the test data
predictions = model.predict(X_test)

# Create DataFrame for submission with PassengerId and predicted 'Survived'
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

# Save predictions to a CSV file
output.to_csv('submission.csv', index=False)

# Print success message
print("Your submission was successfully saved!")
