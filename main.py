# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:38:34 2023

@author: Arsalan Bakhtiar
"""
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the dataset
df = pd.read_csv('Criminal Gangs Dataset.csv')

df.columns

# Remove the 'Area_name' column
df.drop(['Area Name','gang_name','Full Address'], axis=1, inplace=True)


# Create a DataFrame with the given data
data = {
    'City': ['Attock', 'dera ghazi khan', 'faisalabad', 'Jhang'],
    'Keyword': ['Robbery', 'Robbery', 'Car Robbers', 'Robbery']
}

# Define the label encoding mappings for 'City' and 'Keyword'
city_mapping = {city: index for index, city in enumerate(df['City'].unique())}
keyword_mapping = {keyword: index for index, keyword in enumerate(df['Keyword'].unique())}

# Apply label encoding using the map() function
df['City'] = df['City'].map(city_mapping)
df['Keyword'] = df['Keyword'].map(keyword_mapping)


# Drop rows with NaN values
df.dropna(inplace=True)

# Split the dataset into features (X) and target variable (y)
X = df.drop('Keyword', axis=1)  # Replace 'target_variable_name' with the actual column name of the target variable
y = df['Keyword']  # Replace 'target_variable_name' with the actual column name of the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Naive Bayes classifier
naive_bayes = GaussianNB()

# Train the model
naive_bayes.fit(X_train, y_train)

# Make predictions on the test set
y_pred = naive_bayes.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


import pandas as pd
from sklearn.naive_bayes import GaussianNB

# city_mapping = {'Attock': 0, 'dera ghazi khan': 1, 'faisalabad': 2, 'Jhang': 3}


# Create a DataFrame with the given data
data = {
    'City': [0],
    'Latitude': [33.7659684],
    'Longitude': [72.3608754]
}

new_df = pd.DataFrame(data)

# Load the trained Naive Bayes model
naive_bayes = GaussianNB()
naive_bayes.fit(X, y)  # Assuming X and y are the training data used for training the model

# Make predictions on the new data
predictions = naive_bayes.predict(new_df)

# Print the predicted robbery type
print("Predicted Robbery Type:", predictions)

if predictions == 0:
    print('Robbery')
elif predictions == 1:
    print('Car Robbery')
elif predictions == 2:
    print('Motorcycle Robbers')
elif predictions == 3:
    print('Maweshi Dakait')
elif predictions == 4:
    print('Highway Robbers')
elif predictions == 5:
    print('Rahzani')
