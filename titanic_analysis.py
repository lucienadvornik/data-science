import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Načtení trénovacích a testovacích dat
train = pd.read_csv('data/train.csv')  # Zkontroluj cestu k souboru
test = pd.read_csv('data/test.csv')     # Zkontroluj cestu k souboru

# Prohlédnutí prvních 5 řádků trénovacích dat
print(train.head())
print(test.head())

#### Exploratory data analysis ####
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

