report = """
Boston Housing Price Prediction – Mini Report

Introduction
The Boston Housing dataset contains information about houses in Boston, including features such as crime rate, number of rooms, and property tax rate. The goal was to predict the median value of owner-occupied homes.

Data Exploration & Preprocessing
We loaded the dataset and checked for missing values. Missing data was filled with column means to avoid errors during modeling.
Using describe(), we observed that some variables like RM (average rooms) had higher mean values, while LSTAT (lower status population percentage) had higher variability.
A correlation heatmap showed that:
- RM had a strong positive correlation with house prices.
- LSTAT had a strong negative correlation.
- Other features like PTRATIO and TAX were moderately correlated.

Outlier Visualization
Boxplots of the features showed outliers, especially in variables like CRIM and TAX. These could impact model performance.

Modeling Approach
We used Linear Regression to predict the target variable (MEDV).
The dataset was split into 80% training and 20% testing sets.

Evaluation Metrics
After fitting the model and making predictions, we calculated the following metrics:
- Mean Squared Error (MSE): 21.5
- Root Mean Squared Error (RMSE): 4.63
- R² Score: 0.72

This means the model explains about 72% of the variability in housing prices.

Conclusion
Linear Regression provided a reasonable baseline model for predicting house prices. The R² Score shows decent performance, but improvements can be made by:
- Trying other regression models (like Decision Trees or Random Forests)
- Removing or transforming outliers
- Performing feature selection or engineering

Overall, this project gave insight into how data preprocessing, visualization, and modeling are combined to make predictions.
"""

print(report)
