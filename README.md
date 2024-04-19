# Problem Statement

A retail company “ABC Private Limited” wants to understand the customer purchase behavior (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high-volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category), and Total purchase_amount from last month.

Now, they want to build a model to predict the purchase amount of customers against various products which will help them to create personalized offers for customers against different products

# Data Description:

The data is obtained from Kaggle. [Link to the dataset.](https://www.kaggle.com/datasets/sdolezel/black-friday)
The dataset contains two csv files, train.csv and test.csv.
train.csv contains the data to train the model on.
test.csv contains the data to test the model that is trained on the train.csv and has missing values for the target variable.


train.csv is used in this project to perform EDA, Feature Engineering, and to create the model.
The dataset has the following columns:

- User_ID (5891 unique values): This is a unique identifier for each customer in the dataset. There are 5891 distinct customers.
- Product_ID (3631 unique values): This is a unique identifier for each product. There are 3631 different products offered.
- Gender (2 unique values): This column represents the customer's gender as 1 for "Male" and 0 for "Female".
- Age (7 unique values): This column contains the customer's age. There are 7 distinct age groups represented in the data (e.g., 0-17, 18-25, etc.).
- Occupation (21 unique values): This column likely indicates the customer's occupation. There are 21 different occupations represented in the data.
- City_Category (3 unique values): This column categorizes customers based on their city type. There are 3 distinct categories.
- Stay_In_Current_City_Years (5 unique values): This column represents the number of years a customer has resided in their current city. There are 5 distinct durations captured (e.g., 1 year, 2 years, etc.).
- Marital_Status (2 unique values): This column likely indicates the customer's marital status. Similar to gender, it likely contains "Married" and "Single" or similar values.
- Product_Category_1 (20 unique values): This column represents the first level of product categorization. There are 20 distinct categories (e.g., Electronics, Clothing, Appliances).
- Product_Category_2 (17 unique values): This column represents the second level of product categorization, likely more specific than the first level. There are 17 distinct sub-categories.
- Product_Category_3 (15 unique values): This column represents an even more granular third level of product categorization. There are 15 sub-sub-categories present.
- Purchase (18105 value): This column is our target variable indicating the purchase amount of each Black Friday purchase.

# Data Preprocessing

- The data is checked for missing values. We found out that there are 2 columns with missing values.
![Missing Count](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/missing.png)
- The missing values are imputed with the modes of the corresponding columns.
- Product Categories are converted to int datatype from float datatype for efficiency.
- 'User_ID' and 'Product_ID' columns are removed as they are irrelevant for building our machine learning model.
- 'Age' is ordinal encoded to numerical values from 1 to 7 in ascending order, maintaining the hierarchical relationship between age groups.
- 'Stay_In_Current_City_Years' is converted to a int datatype feature by removing the '+' from '4+' leaving the column with values from 1 to 4.

# Exploratory Data Analysis (EDA)

The data is analyzed to understand the relationship between different features and the target variable (purchase amount).
Visualizations are created to explore distributions, correlations, and patterns in the data.


![Missing Count](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/gender.png)
![Purchase Density Plot](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/purchasekde.png)
![City Category Distribution](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/city.png)
![Purchase Outliers](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/purchaseoutliers.png)

Bivariate analysis is performed to examine relationships between features and the target variable.
![Purchase Analysis 1](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/purchase-age.png)
![Purchase Analysis 2](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/purchase-gender.png)
![Purchase Analysis 3](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/purchase-age2.png)
![Purchase Analysis 4](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/purchase-gender2.png)
![Purchase Analysis 5](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/purchaseoccupation.png)

Correlation Analysis is done and the obtained correlation heatmap does not give us any clear insights other than the fact that the features are weakly correlated.
![Correlation heatmap](https://github.com/VishShaji/BlackFriday-EDA-and-Feature-Engineering/blob/main/Assets/corr.png)

# Modeling Approach
Several regression models are trained using the preprocessed data, including Decision Tree, Random Forest, Gradient Boosting, Support Vector Machine (SVM), and a Neural Network.
It should be noted that we have used some algorithms like Linear Regression and Neural Networks that require data that is scaled. Hence we Standardize them using the StandardScaler from SkLearn Library.
Tree-based ensemble algorithms typically do not require scaled data. Decision trees, which serve as the base learners in these algorithms, make splits based on feature values and are not affected by the scale of individual features. Therefore, scaling the data is generally unnecessary for tree-based ensemble methods.

# Code Walkthrough

# Model Evaluation

Root Mean Squared Error (RMSE) is used as the evaluation metric for comparing the performance of different models.
Root Mean Squared Error (RMSE) is a widely used metric for evaluating the performance of regression models. It measures the average magnitude of the errors (the differences between predicted values and actual values) made by the model.
The model with the lowest RMSE is selected as the final model for predicting customer purchase behavior.

# Results and Interpretation

The XGBoost model performs the best among the tested models, with the lowest RMSE. The model can be optimized using GridSearchCV to obtain the best set of parameters for the highest RMSE on the test data.
The optimized XGBoost model further improves performance, providing better predictions of customer purchase behavior.

# Next Steps (Optional)

- Deploy the trained model for making predictions on new data.
- Monitor model performance over time and update as needed.
- Explore additional features or data sources to enhance model accuracy and relevance.
