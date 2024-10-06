#import pandas as pd 
#import seaborn as sns
#import matplotlib.pyplot as plt

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
