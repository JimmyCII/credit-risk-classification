# credit-risk-classification
## Overview of the Analysis

### Purpose of the Analysis
The purpose of this analysis is to develop and evaluate machine learning models to predict credit risk. Specifically, we aim to classify loans as either healthy (low-risk) or high-risk based on various financial attributes of the applicants.

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
   - Selected the `LogisticRegression` algorithm from the `sklearn` library for its simplicity and effectiveness in binary classification tasks.
   - Instantiated the logistic regression model with a `random_state` parameter of 1 to ensure reproducibility.

3. **Model Training:**
   - Trained the logistic regression model using the training data (`X_train` and `y_train`).

4. **Model Evaluation:**
   - Made predictions on the test data (`X_test`).
   - Evaluated the model's performance using accuracy, precision, recall, and the confusion matrix.
   - Generated a classification report to summarize the model's performance metrics.

### Methods Used
- **Logistic Regression:** A linear model used for binary classification tasks. It estimates the probability that a given input belongs to a particular class.
- **Accuracy Score:** Measures the proportion of correctly classified instances out of the total instances.
- **Precision and Recall:** Precision measures the accuracy of positive predictions, while recall measures the ability to identify all positive instances.
- **Confusion Matrix:** A table that summarizes the performance of a classification model by showing the counts of true positives, true negatives, false positives, and false negatives.

## Results


* **Logistic Regression Model:**
    * **Overall Accuracy:** 98.7%
    * **Healthy Loan (0) Accuracy:** 94.2%
    * **High-Risk Loan (1) Accuracy:** 99.4%
    * **Precision for Healthy Loans (0):** 94.2%
    * **Recall for Healthy Loans (0):** 84.1%
    * **Precision for High-Risk Loans (1):** 99.4%
    * **Recall for High-Risk Loans (1):** 99.8%
    * **Confustion Matrix:**

        ![Confusion Matrix](images/confusion_matrix.png)



## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any.

* **Best Performing Model:**
  - The logistic regression model performs exceptionally well, with an overall accuracy of 98.7%.
  - It has high precision and recall for high-risk loans, making it highly effective at identifying high-risk loans.

* **Performance Considerations:**
  - The model's performance is crucial for predicting high-risk loans (`1`), as it has a recall of 99.8%, ensuring that almost all high-risk loans are correctly identified.
  - The precision for healthy loans (`0`) is also high at 94.2%, indicating that the model is reliable in predicting healthy loans.

* **Recommendation:**
  - Based on the evaluation metrics, the logistic regression model is recommended for predicting credit risk. Its high accuracy, precision, and recall make it suitable for identifying both healthy and high-risk loans effectively.

If you do not recommend any of the models, please justify your reasoning.