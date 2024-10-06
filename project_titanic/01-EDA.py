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
