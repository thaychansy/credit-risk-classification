# Logistic Regression Model for Loan Prediction

This repository contains a Jupyter notebook that demonstrates the process of training and evaluating a Logistic Regression model to predict loan status (healthy vs. high-risk) using a dataset of loan data.

## Contents

- **Overview of the Analysis**: A description of the machine learning analysis conducted, including the purpose, data, and process.
- **Importing Libraries**: Import necessary modules including `numpy`, `pandas`, and `scikit-learn`.
- **Data Preparation**: Load the dataset, and split it into features and labels.
- **Data Splitting**: Split the dataset into training and testing sets using `train_test_split`.
- **Model Training**: Train a Logistic Regression model using the training data.
- **Model Evaluation**: Evaluate the model's performance by generating a confusion matrix and classification report.
- **Interpretation**: Analyze the results to determine how well the model predicts loan statuses.
- **Results**: A summary of the performance metrics for the machine learning models.
- **Summary**: A final summary of the analysis, including recommendations.

## Overview of the Analysis

In this analysis, the objective was to build and evaluate a machine learning model to predict the likelihood of a loan being healthy or high-risk. The dataset contained financial information about loans, including variables such as loan size, interest rate, borrower income, and debt-to-income ratio.

The target variable we needed to predict was the `loan_status`, which indicates whether a loan is healthy (`0`) or high-risk (`1`). The analysis followed the typical stages of the machine learning process:

- **Data Preparation**: We began by loading the data and splitting it into features (`X`) and labels (`y`).
- **Data Splitting**: We then split the data into training and testing sets to allow for model training and evaluation.
- **Model Training**: A Logistic Regression model was used as the primary algorithm to predict loan status.
- **Model Evaluation**: The model's performance was evaluated using accuracy, precision, and recall metrics to understand its effectiveness in predicting both healthy and high-risk loans.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook (optional, for running the notebook interactively)

## Usage

1. **Importing Libraries**:
    - Start by importing the required libraries:
      ```python
      import numpy as np
      import pandas as pd
      from pathlib import Path
      from sklearn.metrics import confusion_matrix, classification_report
      ```

2. **Load and Prepare Data**:
    - Load the loan dataset from a CSV file into a Pandas DataFrame:
      ```python
      df_lending_data = pd.read_csv("Resources/lending_data.csv")
      ```

    - Separate the data into features (`X`) and labels (`y`):
      ```python
      y = df_lending_data['loan_status']
      X = df_lending_data.drop(columns=['loan_status'])
      ```

3. **Split Data**:
    - Split the dataset into training and testing sets:
      ```python
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
      ```

4. **Train Model**:
    - Instantiate and train the Logistic Regression model:
      ```python
      from sklearn.linear_model import LogisticRegression
      model = LogisticRegression(max_iter=200, random_state=1)
      model.fit(X_train, y_train)
      ```

5. **Make Predictions**:
    - Generate predictions on the test dataset:
      ```python
      predictions = model.predict(X_test)
      ```

6. **Evaluate Model**:
    - Generate and display a confusion matrix:
      ```python
      cm = confusion_matrix(y_test, predictions)
      cm_df = pd.DataFrame(
          cm, index=["Actual Healthy Loan 0", "Actual High-Risk Loan 1"], 
          columns=["Predicted Healthy Loan 0", "Predicted High-Risk Loan 1"]
      )
      print("Confusion Matrix")
      display(cm_df)
      ```

7. **Results Interpretation**:
    - Analyze how well the logistic regression model predicts both the `0` (healthy loan) and `1` (high-risk loan) labels based on the confusion matrix.

## Results

Below are the accuracy, precision, and recall scores for the Logistic Regression model used in this analysis:

- **Logistic Regression Model**:
  - **Accuracy**: 0.9926
  - **Precision**: 0.8447
  - **Recall**: 0.9402

  

## Summary

The Logistic Regression model showed strong performance in predicting loan statuses, with high accuracy, precision, and recall scores. Based on the results, this model is recommended for use, particularly because:

- **Performance**: The model performs well across both healthy and high-risk loan predictions, with balanced precision and recall scores.
- **Problem-Specific Consideration**: If the goal is to minimize financial risk, predicting high-risk loans accurately is crucial. The model’s high recall for predicting high-risk loans suggests it is effective in this regard.

Overall, the Logistic Regression model is a reliable choice for predicting loan status. However, depending on the specific needs (e.g., focusing more on minimizing false negatives), further tuning or exploring alternative models may be beneficial.
