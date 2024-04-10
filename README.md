# Problem Statement

A retail company “ABC Private Limited” wants to understand the customer purchase behavior (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high-volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category), and Total purchase_amount from last month.

Now, they want to build a model to predict the purchase amount of customers against various products which will help them to create personalized offer for customers against different products

# Data Description:

The data is obtained from Kaggle. [Link to the dataset.](https://www.kaggle.com/datasets/sdolezel/black-friday)
The dataset contains two csv files, train.csv and test.csv.
train.csv contains the data to train the model on.
test.csv contains the data to test the model that is trained on the train.csv and has missing values for the target variable.


train.csv is used in this project to perform EDA, Feature Engineering and to create the model.
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

# Exploratory Data Analysis (EDA)

# Modeling Approach

# Code Walkthrough

# Model Evaluation

# Results and Interpretation

# Next Steps (Optional)

