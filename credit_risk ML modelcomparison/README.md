# Credit Risk Classification Model Comparison

## Overview of the Analysis

### Purpose of the Analysis
The purpose of this analysis is to develop and evaluate multiple machine learning models to predict credit risk. Specifically, we aim to classify loans as either healthy (low-risk) or high-risk based on various financial attributes of the applicants.

### Financial Information and Prediction Objective
The dataset contains financial information about loan applicants, including features such as loan amount, income, debt-to-income ratio, and credit history. The target variable is a binary label indicating whether a loan is high-risk (`1`) or healthy (`0`). The objective is to accurately predict this target variable using the provided features.

### Basic Information about the Variables
The target variable distribution is as follows:
- Healthy loans (`0`): 18,691 instances
- High-risk loans (`1`): 619 instances

### Stages of the Machine Learning Process
1. **Data Preprocessing:**
   - Loaded the dataset and performed exploratory data analysis (EDA) to understand the data distribution and identify any missing values.
   - Cleaned the data by handling missing values and encoding categorical variables.
   - Split the data into training and testing sets to evaluate model performance.

2. **Model Development:**
   - Selected multiple machine learning algorithms for comparison.
   - Instantiated each model with appropriate parameters.

3. **Model Training:**
   - Trained each model using the training data (`X_train` and `y_train`).

4. **Model Evaluation:**
   - Made predictions on the test data (`X_test`).
   - Evaluated each model's performance using accuracy, precision, recall, and the confusion matrix.
   - Generated classification reports to summarize each model's performance metrics.

## Methods Used

### Logistic Regression
- **Description:** Logistic Regression is a linear model used for binary classification tasks. It estimates the probability that a given input belongs to a particular class.
- **Best Uses:** Logistic Regression is best used for binary classification problems where the relationship between the features and the target variable is approximately linear.
- **Results:**
  - **Overall Accuracy:** 99.2%
  - **Healthy Loan (0) Accuracy:** 99.4%
  - **High-Risk Loan (1) Accuracy:** 94.2%
  - **Precision for Healthy Loans (0):** 100%
  - **Recall for Healthy Loans (0):** 99.4%
  - **Precision for High-Risk Loans (1):** 84%
  - **Recall for High-Risk Loans (1):** 94%
  - **Confusion Matrix:**
    ![Logistic Regression Confusion Matrix](images/cm_logistic_regression.png)

### Decision Tree
- **Description:** Decision Tree is a non-linear model that splits the data into subsets based on the value of input features. It is easy to interpret and visualize.
- **Best Uses:** Decision Trees are best used for classification and regression tasks where interpretability is important, and the relationship between features and the target variable is non-linear.
- **Results:**
  - **Overall Accuracy:** 99.0%
  - **Healthy Loan (0) Accuracy:** 99.4%
  - **High-Risk Loan (1) Accuracy:** 85.2%
  - **Precision for Healthy Loans (0):** 100%
  - **Recall for Healthy Loans (0):** 99.5%
  - **Precision for High-Risk Loans (1):** 84%
  - **Recall for High-Risk Loans (1):** 85%
  - **Confusion Matrix:**
    ![Decision Tree Confusion Matrix](images/cm_decision_tree.png)

### Random Forest
- **Description:** Random Forest is an ensemble model that combines multiple decision trees to improve accuracy and reduce overfitting.
- **Best Uses:** Random Forest is best used for classification and regression tasks where high accuracy is required, and the relationship between features and the target variable is complex.
- **Results:**
  - **Overall Accuracy:** 99.1%
  - **Healthy Loan (0) Accuracy:** 99.5%
  - **High-Risk Loan (1) Accuracy:** 89.3%
  - **Precision for Healthy Loans (0):** 100%
  - **Recall for Healthy Loans (0):** 99.5%
  - **Precision for High-Risk Loans (1):** 85%
  - **Recall for High-Risk Loans (1):** 89%
  - **Confusion Matrix:**
    ![Random Forest Confusion Matrix](images/cm_random_forest.png)

### Support Vector Machine (SVM)
- **Description:** SVM is a linear model that finds the hyperplane that best separates the classes in the feature space.
- **Best Uses:** SVM is best used for binary classification tasks with high-dimensional data and clear margin of separation between classes.
- **Results:**
  - **Overall Accuracy:** 99.4%
  - **Healthy Loan (0) Accuracy:** 99.4%
  - **High-Risk Loan (1) Accuracy:** 98.9%
  - **Precision for Healthy Loans (0):** 100%
  - **Recall for Healthy Loans (0):** 99.4%
  - **Precision for High-Risk Loans (1):** 84%
  - **Recall for High-Risk Loans (1):** 99%
  - **Confusion Matrix:**
    ![SVM Confusion Matrix](images/cm_svm.png)

### K-Nearest Neighbors (KNN)
- **Description:** KNN is a non-parametric model that classifies instances based on the majority class of their k-nearest neighbors.
- **Best Uses:** KNN is best used for classification tasks with small to medium-sized datasets and where the decision boundary is non-linear.
- **Results:**
  - **Overall Accuracy:** 99.3%
  - **Healthy Loan (0) Accuracy:** 99.4%
  - **High-Risk Loan (1) Accuracy:** 97.4%
  - **Precision for Healthy Loans (0):** 100%
  - **Recall for Healthy Loans (0):** 99.4%
  - **Precision for High-Risk Loans (1):** 84%
  - **Recall for High-Risk Loans (1):** 97%
  - **Confusion Matrix:**
    ![KNN Confusion Matrix](images/cm_knn.png)

### Gradient Boosting
- **Description:** Gradient Boosting is an ensemble model that builds multiple weak learners (usually decision trees) sequentially to minimize the loss function.
- **Best Uses:** Gradient Boosting is best used for classification and regression tasks where high accuracy is required, and the relationship between features and the target variable is complex.
- **Results:**
  - **Overall Accuracy:** 99.4%
  - **Healthy Loan (0) Accuracy:** 99.4%
  - **High-Risk Loan (1) Accuracy:** 98.9%
  - **Precision for Healthy Loans (0):** 100%
  - **Recall for Healthy Loans (0):** 99.4%
  - **Precision for High-Risk Loans (1):** 84%
  - **Recall for High-Risk Loans (1):** 99%
  - **Confusion Matrix:**
    ![Gradient Boosting Confusion Matrix](Images\cm_gradient_boosting.png)

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any.

* **Best Performing Model:**
  - Based on the evaluation metrics, the Logistic Regression model performs exceptionally well, with an overall accuracy of 99.2%.
  - It has high precision and recall for high-risk loans, making it highly effective at identifying high-risk loans.

* **Performance Considerations:**
  - The model's performance is crucial for predicting high-risk loans (`1`), as it has a recall of 94%, ensuring that almost all high-risk loans are correctly identified.
  - The precision for healthy loans (`0`) is also high at 100%, indicating that the model is reliable in predicting healthy loans.

* **Recommendation:**
  - Based on the evaluation metrics, the Logistic Regression model is recommended for predicting credit risk. Its high accuracy, precision, and recall make it suitable for identifying both healthy and high-risk loans effectively.

