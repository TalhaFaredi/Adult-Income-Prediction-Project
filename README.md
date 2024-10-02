# README.md

# Adult Income Prediction Project

This project utilizes the Adult Income dataset from the UCI Machine Learning Repository to predict whether an individual's income exceeds $50K per year based on various attributes. The dataset contains personal information such as age, work class, education, occupation, marital status, and other socioeconomic factors.

## Project Overview

The primary objective of this project is to apply machine learning techniques to predict income levels (`<=50K` or `>50K`) using a K-Nearest Neighbors (KNN) classifier. The dataset is preprocessed to handle missing values, categorical data encoding, and feature engineering to improve model performance.

## Dataset

The dataset includes 15 attributes:

- Age
- Workclass
- Fnlwgt
- Education
- Education-num
- Marital-status
- Occupation
- Relationship
- Race
- Sex
- Capital-gain
- Capital-loss
- Hours-per-week
- Native-country
- Income (Target: `<=50K` or `>50K`)

## Key Steps

1. **Data Preprocessing**: 
   - Missing values are handled by removing irrelevant columns like `education-num` (which overlaps with the `education` column), `fnlwgt`, `race`, and financial gain/loss.
   - The `age` and `hours-per-week` features are used to create new columns like `Seniority` (age groups: Young, Middle Age, Senior) and `Work Type` (hours worked: Part-Time, Full-Time, Over-Time).
   - Categorical columns like `marital-status`, `relationship`, and `sex` are combined to form a single column for dimensionality reduction.

2. **Feature Engineering**:
   - New columns like `Seniority` and `Work Type` are created based on specific conditions.
   - Merging of `marital-status`, `relationship`, and `sex` into a new feature for simplification.

3. **Label Encoding**:
   - All categorical variables are encoded into numerical values using `LabelEncoder`.

4. **Model Training**:
   - K-Nearest Neighbors (KNN) is employed as the machine learning algorithm for classification.
   - The model is trained with 35 neighbors, chosen to balance between model complexity and performance.

5. **Model Testing**:
   - The test dataset undergoes the same preprocessing steps as the training data.
   - Predictions are made using the trained KNN model.
   - Evaluation metrics like the confusion matrix and accuracy score are calculated to assess the performance.

## Results

- **Confusion Matrix**: A heatmap of the confusion matrix is generated to visually represent the classification results.
- **Accuracy**: The model achieved an accuracy score of approximately 82.5%, indicating a good performance on the given dataset.

## Tools and Libraries

- **Programming Language**: Python
- **Libraries**: 
  - NumPy & Pandas for data manipulation
  - Scikit-learn for machine learning (KNN, Label Encoding, Accuracy, Confusion Matrix)
  - Matplotlib & Seaborn for data visualization

## Conclusion

The project successfully predicts income categories based on the given features with an accuracy of 82.5%. The preprocessing techniques, such as feature engineering and label encoding, played a vital role in improving model performance. Future improvements may involve trying other machine learning models, tuning hyperparameters, and expanding the feature set.


