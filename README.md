# Crime Type Prediction Using Machine Learning

This repository contains a Python script for predicting crime types using a machine learning approach with a focus on the Naive Bayes classifier. The dataset utilized in this analysis contains valuable information about criminal activities, with detailed records of different types of crimes reported in various cities, along with corresponding latitude and longitude coordinates.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The objective of this project is to leverage machine learning techniques to predict the type of crime based on location-related features. The selected model for this prediction task is the Gaussian Naive Bayes classifier, which can effectively handle categorical data and make probabilistic predictions.

## Dataset
The dataset used in this project is available in the 'Criminal Gangs Dataset.csv' file. It includes various attributes such as 'City', 'Keyword' (crime type), 'Latitude', and 'Longitude', providing essential information for training and evaluating the predictive model.

## Installation
To run the script, ensure you have Python installed. You can clone this repository using the following command:
```
https://github.com/arsalanbakhtiarab/Crime_Type_Prediction_Using_Machine_Learning.git
```

Next, navigate to the project directory and install the required dependencies:
```
pip install pandas scikit-learn
```

## Usage
1. Preprocess the data: The script will automatically load and preprocess the dataset, converting categorical features into numerical representations using label encoding.

2. Model Training: The Gaussian Naive Bayes classifier will be used to train the model on the prepared data, learning patterns and relationships between features and crime types.

3. Model Evaluation: The trained model's accuracy will be assessed on a test set to measure its performance in crime type prediction.

4. Crime Type Prediction: After successful training, the script can be utilized to predict the most probable crime type based on new data points, including 'City', 'Latitude', and 'Longitude' information.

## Results
The project aims to provide insights into criminal activities across different cities and demonstrate the effectiveness of machine learning in predicting crime types. The accuracy of the trained Naive Bayes classifier will be reported to evaluate its performance.

## Contributing
Contributions to this project are welcome! If you have any suggestions, bug fixes, or improvements, please feel free to open an issue or submit a pull request.
