# Loan Default Prediction Project by Team Z

## Overview
This project focuses on predicting loan defaults based on various customer attributes using machine learning. The dataset consists of information related to individuals' loan applications, such as income, loan amount, credit score, and employment status, with the goal of predicting whether a borrower will default on a loan.


## Competition

This project is based on the [Loan Default Prediction Competition](https://www.kaggle.com/competitions/lend-or-lose) on Kaggle. The objective of the competition is to predict loan defaults based on customer data. 

The models developed in this project use the same dataset provided by the competition and are evaluated based on similar metrics, such as accuracy and other performance metrics.

For more details on the competition, please visit the link above.

## Problem Statement
The problem at hand is to classify loan applicants into two categories: defaulters (those who will not repay the loan) and non-defaulters. The dataset contains features like income, credit score, loan amount, and other demographic information. The challenge is to build a model that accurately predicts loan default based on these features, considering the relatively low defaulter rate (≤ 5%).

## Notebook

This repository includes a Jupyter notebook (`Loan_Default_Prediction.ipynb`) that demonstrates all the steps performed in the project. The notebook includes:

- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Hyperparameter tuning
- Comparison of various machine learning models

You can run the notebook to explore the complete workflow and gain insights into how the loan default prediction model was built and optimized.

## Datasets

This repository contains the following datasets:

- `train.csv`: The training dataset used for building and training the machine learning models. It contains information about loan applicants, including features like income, credit score, loan amount, and more.
- `test.csv`: The test dataset used for evaluating the performance of the trained models. This dataset contains similar features but with the target variable (loan default) not included for prediction.

Both datasets are used to demonstrate the loan default prediction workflow, with the `train.csv` used for model training and the `test.csv` used for testing and evaluation.

## Report

This repository also includes a detailed report that specifies all the models used in the project, along with their accuracy and the procedures followed during the model selection process. The report outlines the steps taken for data preprocessing, feature engineering, model evaluation, and hyperparameter tuning, providing a comprehensive understanding of the approach to loan default prediction.

Feel free to refer to the report for more in-depth information.

## Data Preprocessing
- **Cleaning**: Checked for null values, duplicates, and outliers (none found).
- **Feature Encoding**: Categorical features were label-encoded for compatibility with machine learning models.
- **Feature Selection**: A correlation matrix was used to identify the most relevant features, resulting in the exclusion of irrelevant columns.

## Exploratory Data Analysis (EDA)
EDA revealed key patterns and insights:
- **Outliers**: No significant outliers were detected.
- **Defaulter Trends**: Defaulters were typically younger, had lower incomes, higher loan amounts, and higher interest rates compared to non-defaulters.

## Model Selection
Multiple machine learning models were tested to identify the best-performing one. The models considered included:
- Decision Trees
- Random Forest
- Logistic Regression
- AdaBoost
- SVM
- Artificial Neural Networks (ANN)

## Evaluation Metric
- **K-Fold Cross-Validation**: Used to assess the models' generalizability and avoid overfitting.
- **Performance Comparison**: Models were compared based on accuracy, with **XGBoost** achieving the highest performance.

## Final Model: XGBoost
XGBoost was selected as the final model due to its ability to handle mixed data types (numerical and categorical) and its robustness against overfitting. The model achieved a Kaggle score of **0.88789**.

### Hyperparameter Tuning
Optimized using `RandomizedSearchCV` to fine-tune the following parameters:
```python
{'objective': 'binary:logistic', 'eval_metric': 'mlogloss',
 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100,
 'random_state': 67, 'subsample': 0.6, 'scale_pos_weight': 1,
 'use_label_encoder': False}
```
## Insights
- **Defaulter Characteristics**: Key characteristics of defaulters included lower income, younger age, higher loan amounts, and higher interest rates. These patterns were identified during exploratory data analysis and were consistent across various models.
  
- **Feature Engineering**: Adding noise for skewing and introducing synthetic features did not result in significant improvements in model performance. In some cases, adding noise led to overfitting, where the model performed well on the training data but struggled to generalize on the test data. Therefore, these techniques were not included in the final model development.

  ## Conclusion
The XGBoost model performed the best with an accuracy of 0.88789, making it the optimal choice for predicting loan defaults in this dataset. Its superior performance compared to other models such as Decision Trees, Random Forest, and SVM can be attributed to its robust handling of mixed data types and its ability to generalize well, even with the challenges posed by the dataset's characteristics.

## Key Takeaways
- **Data Preprocessing and Feature Selection**: These were crucial steps in enhancing the model's performance. Properly handling missing values, encoding categorical variables, and selecting the most relevant features significantly improved the model's accuracy.
  
- **Handling Class Imbalance**: The low defaulter rate (≤ 5%) posed a challenge in balancing the dataset, but XGBoost's ability to handle this imbalance (using the `scale_pos_weight` parameter) played a key role in achieving high predictive performance.
  
- **Model Choice**: While multiple models were tested, XGBoost's combination of speed, accuracy, and scalability made it the best choice for this problem.


This README is structured to be informative, highlighting the core aspects of the project, from data preprocessing to model selection and evaluation. Feel free to copy and modify this according to your specific needs.
