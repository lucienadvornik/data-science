import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk('/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')
df.head()

# Inspection of first rows
print(df.head())

# Basic information about the dataset
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()

# Remove rows with missing values in the 'review_text' column
df_clean = df.dropna(subset=['review_text'])

# Remove duplicate entries
df_clean = df_clean.drop_duplicates()

# Convert the 'review_date' column to the correct date format
df_clean['review_date'] = pd.to_datetime(df_clean['review_date'])

# Display the cleaned data information
print(df_clean.info())
