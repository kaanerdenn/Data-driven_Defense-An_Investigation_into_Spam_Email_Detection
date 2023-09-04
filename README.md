# Data-driven_Defense-An_Investigation_into_Spam_Email_Detection

This documentation summarizes the process of email spam classification using Python, Scikit-Learn, and Logistic Regression.

## Task 1: Data Loading and Overview

We start by loading email data from a CSV file named 'mail_data.csv' and print the raw data.

## Task 2: Handling Null Values

We replace null values in the dataset with empty strings for data consistency.

## Task 3: Labeling Spam and Ham

We label spam emails as 0 and ham (non-spam) emails as 1.

## Task 4: Data Preparation

We separate the data into text messages (X) and labels (Y) for classification.

## Task 5: Train-Test Split

We split the data into training and testing sets using a 80-20 split ratio.

## Task 6: Text Feature Extraction

We use TF-IDF vectorization to transform text data into feature vectors for input to the Logistic Regression model.

## Task 7: Model Training

We train a Logistic Regression model on the training data.

## Task 8: Model Evaluation on Training Data

We evaluate the model's performance on the training data and calculate accuracy, precision, recall, and F1-score.

## Task 9: Model Evaluation on Test Data

We evaluate the model's performance on the test data and calculate accuracy, precision, recall, and F1-score.

## Task 10: Cross-Validation

We perform cross-validation to assess the model's performance and calculate the mean cross-validation score.

## Task 11: Building a Predictive System

We demonstrate how to make predictions on new email text and determine whether it's spam or ham.

## Task 12: Performance Metrics

We provide a classification report and confusion matrix to evaluate the model's overall performance.

---

This documentation provides an overview of the email spam classification process, including data loading, preprocessing, model training, evaluation, and building a predictive system. You can adapt and expand upon this document as needed for your project documentation.
