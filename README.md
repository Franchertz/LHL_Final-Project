# Project Name : Credit Card Fraud Detection using PyCaret

In this project, we will initially create predictions using traditional Machine Learning techniques. Subsequently, we will employ AutoML (Automated Machine Learning) techniques to achieve the same goal. Throughout this notebook, we will leverage PyCaret for our analysis.

# Dataset Description:

This dataset comprises credit card transactions made by European cardholders in September 2013. Over a span of two days, it records a total of 284,807 transactions, among which 492 are classified as fraudulent. The dataset exhibits a significant class imbalance, with fraudulent transactions accounting for only 0.172% of the total.

The dataset consists exclusively of numerical input variables resulting from Principal Component Analysis (PCA). Unfortunately, due to confidentiality constraints, we cannot disclose the original features or provide additional context about the data. The features V1 through V28 represent the principal components obtained through PCA, while 'Time' and 'Amount' are the only features not subjected to PCA transformation. 'Time' indicates the time elapsed in seconds since the first transaction in the dataset, and 'Amount' signifies the transaction amount. The 'Amount' feature can be valuable for techniques such as cost-sensitive learning. The 'Class' feature serves as the response variable, taking a value of 1 for fraudulent transactions and 0 for legitimate ones.

Due to the class imbalance, it is advisable to assess model performance using the Area Under the Precision-Recall Curve (AUPRC), as accuracy based on the confusion matrix may not be meaningful for unbalanced classification problems.

# Problem Statement:

In today's digital landscape, the rise of electronic transactions has given way to an increasingly concerning issue – credit card fraud. Criminals are continually exploring new avenues to exploit vulnerabilities within payment systems, resulting in significant financial losses for both individuals and businesses. As a result, the development of robust fraud detection mechanisms has become paramount.

Credit card fraud detection leverages cutting-edge technologies and advanced algorithms to identify and thwart fraudulent activities in real-time. By scrutinizing a multitude of data points and patterns, including transaction history, spending habits, and geographic locations, these systems can effectively flag suspicious transactions and take immediate action to mitigate potential financial losses.

Detecting credit card fraud is an intricate and ever-evolving discipline, demanding ongoing innovation and adaptability to outmaneuver sophisticated fraudulent tactics. Machine learning and artificial intelligence serve as pivotal tools within this domain, facilitating the creation of intelligent systems capable of learning from past fraudulent behaviors and swiftly identifying emerging fraud trends.

# Objective
- The dataset primarily comprises a minuscule fraction of transactions categorized as fraudulent. Our goal is to accurately identify and isolate these fraudulent transactions within the dataset.

- Leveraging the available data, our aim is to derive a set of actionable insights and recommendations that can assist the credit card company in implementing effective measures to prevent unwarranted charges for their customers.


### Table of Contents

1. [Loading all Required Libraries](#1--Loading all Required Libraries)
2. [Load Data](#2)
3. [EDA (Exploratory Data Analysis)](#3)
   - 3.1 [Handling Missing Values](#3.1)
   - 3.2 [Correlation between Features](#3.2)
   - 3.3 [Outliers](#3.3)
   - 3.4 [Skewness and Kurtosis](#3.4)
4. [Data Preprocessing](#4)
   - 4.1 [Handling Imbalanced Data](#4.1)
5. [Model Building and Evaluation](#5)
   - 5.1 [Logistic Regression](#5.1)
     - 5.1.1 [Train the Model without Handling the Imbalanced Class Distribution](#5.1.1)
     - 5.1.2 [Imbalanced Data Handling Techniques](#5.1.2)
       - a) [SMOTE](#a)
       - b) [Near Miss](#b)
       - c) [Random Under Sample](#c)
   - 5.2 [XGBoost Classifier](#5.2)
6. [Using PyCaret To Detect Credit Card Fraud](#6)

# STEPS:

These steps collectively form a comprehensive process for analyzing credit card transaction data, addressing class imbalance, building machine learning models, and automating the process using PyCaret to enhance efficiency and model evaluation.


1. Loading all Required Libraries:
   - In this section, we will import the necessary Python libraries and modules that will be used throughout the analysis. These libraries provide various functions and tools to work with data and build machine learning models.

2. Load Data:
   - In this step, we will load the dataset containing credit card transaction data for analysis. The dataset represents transactions made by European cardholders in September 2013.

3. EDA (Exploratory Data Analysis):
   - This section focuses on understanding the dataset through exploratory data analysis.
   - Handling Missing Values: We will check and handle any missing values in the dataset.
   - Correlation between Features: We'll examine the correlations between the dataset's features to identify potential relationships.
   - Outliers: Detection and treatment of outliers in the data to ensure they don't affect model performance.
   - Skewness and Kurtosis: We'll analyze the skewness and kurtosis of the data distribution, which helps us understand the data's shape.

4. Data Preprocessing:
   - In this section, we will prepare the data for model building.
   - Handling Imbalanced Data: Address the issue of class imbalance by implementing techniques to handle the minority class (fraudulent transactions).

5. Model Building and Evaluation:
   - This section involves constructing machine learning models and evaluating their performance.
   - Logistic Regression: We'll start with a logistic regression model.
     - Training the Model without Handling the Imbalanced Class Distribution: Build a logistic regression model without addressing class imbalance.
     - Imbalanced Data Handling Techniques: Explore techniques like SMOTE, Near Miss, and Random Under Sample to deal with class imbalance.
   - XGBoost Classifier: Implement an XGBoost classifier to compare its performance.

6. Using PyCaret for Credit Card Fraud Detection
   - In this final section, we'll leverage the PyCaret library to streamline the machine learning workflow and automate model selection and evaluation for credit card fraud detection.


## Conclusion

Comparison and Summary:

In the analysis, two different approaches to building and evaluating machine learning models for credit card fraud detection are discussed. Let's summarize the findings and compare these two methods:

**Traditional Method:**

1. **Imbalanced Dataset Mitigation:**
   - This method uses traditional techniques like SMOTE, Near Miss, and Random Under Sample to handle class imbalance.

2. **Model Evaluation and Recommendations:**
   - The XGBoost model also performs well, with high accuracy, precision, recall, and F1 score. It is recommended for deployment.

   Accuracy XGB: 0.9996
   Precision XGB: 0.944
   Recall XGB: 0.803
   F1 Score XGB: 0.868

**Using PyCaret Method:**

   **Why use PyCaret?**
  - PyCaret helps with data preprocessing, training multiple models simultaneously, and outputs a table comparing model performance.

   **Result Summary:**
   - PyCaret compared and selected the best model based on precision, recall, F1-score, and accuracy.
   - The best model identified was the RandomForestClassifier.
   - The model achieved high accuracy and AUC values across all cross-validation folds, indicating excellent classification performance.
   - The recall values range from approximately 0.725 to 0.853 in individual folds, indicating a reasonable ability to detect fraud.
   - Precision values are generally high, with an average precision of approximately 94.8%, indicating accurate positive predictions.
   - The F1 score, a balance between precision and recall, has a mean of approximately 86.5%, indicating a good balance between accurate positive predictions and fraud detection.
   - Kappa and MCC values suggest a substantial level of agreement between model predictions and actual data, even after considering random chance.

**Comparison and Conclusion:**

The traditional method employs explicit techniques for class imbalance mitigation and showcases the performance of individual models like Random Forest and XGBoost. It provides a comprehensive evaluation and recommendation.

On the other hand, PyCaret simplifies the model comparison process and chooses the RandomForestClassifier as the best model based on multiple metrics. It offers a more streamlined approach to model selection and performance evaluation.

Both methods yield models with high accuracy, precision, and recall, making them suitable for the task of credit card fraud detection. The choice between the two methods may depend on the complexity of the dataset and the preferred level of control over the modeling process. PyCaret offers a more automated and user-friendly approach, while the traditional method allows for fine-tuning and customization.

