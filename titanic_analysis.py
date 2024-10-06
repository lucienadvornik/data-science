import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# Loading training and test data
train = pd.read_csv('data/train.csv')  # Check the file path
test = pd.read_csv('data/test.csv')    # Check the file path

# Viewing the first 5 rows of the training data
print(train.head())
print(test.head())


#### 1) EXPLORATORY DATA ANALYSIS ####
# Basic statistics
print(train.describe())

# Check for missing values
print(train.isnull().sum())

# Distribution of survival
print(train['Survived'].value_counts())

# Visualization of survival by gender
sns.countplot(x='Survived', hue='Sex', data=train)
plt.title('Survival by Gender')          # Title of the plot
plt.show()                                # Show the plot

# Visualization of survival by age
sns.histplot(data=train, x='Age', hue='Survived', multiple='stack', kde=False)
plt.title('Survival by Age')             # Title of the plot
plt.show()                                # Show the plot

# Visualization of survival by passenger class
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('Survival by Passenger Class')  # Title of the plot
plt.show()                                # Show the plot


#### 2) DATA PREPROCESSING ####
# Check for missing values 
print(train.isnull().sum())

# Fill missing Age with the mean age
train['Age'].fillna(train['Age'].mean(), inplace=True)

# Check for missing values again
print(train.isnull().sum())

##
# Encode gender (male = 0, female = 1)
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})

# Encode Embarked (C = 0, Q = 1, S = 2)
train['Embarked'].fillna('S', inplace=True)  # Fill missing values in 'Embarked'
train['Embarked'] = train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})


#### 3) FEATURE SELECTION ####
# Select features and target variable
X = train[['Pclass', 'Sex', 'Age', 'Fare']]  # Features used for training
y = train['Survived']                         # Target variable


#### 4) TRAIN-TEST SPLIT ###
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#### 5) MODEL TRAINING ####
from sklearn.linear_model import LogisticRegression
# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)


#### 6) MODEL EVALUATION ####
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
