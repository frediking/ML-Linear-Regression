# California Housing Price Prediction

## Overview

This project builds a linear regression model to predict housing prices in California based on socioeconomic and geographical factors. Using Scikit-learn’s California Housing Dataset, we analyze how features like median income, house age, and location influence housing prices. The goal is to develop an interpretable model that provides insights into real estate trends.

## Problem Statement

The cost of housing is a crucial factor affecting economic stability and quality of life. Accurately predicting housing prices can help policymakers, real estate developers, and buyers make informed decisions. This project applies linear regression to predict the median house value (target) using features such as median income, house age, number of rooms, population, and geographical location . The model’s performance is evaluated using Mean Squared Error (MSE) to assess prediction accuracy.

## Dataset

The California Housing Dataset is sourced from Scikit-learn and contains 20,640 samples with the following features :

1. MedInc :Median income of the neighborhood
2. HouseAge: Average age of houses in the area
3. AveRooms: Average number of rooms per household
4. AveBedrms: Average number of bedrooms per household
5. Population: Population of the neighborhood
6. AvgOccup: Average household occupancy
7. Latitude & Longitude: Geographical coordinates



The TARGET VARIABLE is MedHouseVal(median house value in $100,000s).

## Feature Scaling & Transformation (Did It Help?!)

Why I Scaled & Transformed Features

Feature scaling is often used in machine learning to ensure that numerical values across different features contribute equally to model training. 
We applied: 
1. Log Transformation to MedInc and Population to normalize skewed distributions.

2. Standardization (Z-score Scaling) to ensure features have zero mean and unit variance.

### MSE Results (Before & After Scaling)

| Model         | Features Used       | Mean Squared Error (MSE) |
|--------------|--------------------|--------------------------|
| Linear Reg.  | Raw Features       | 0.5559                   |
| Linear Reg.  | Scaled Features    | 0.5899                   |

## Why Scaling Didn’t Help

•**Linear regression is less sensitive to feature scaling** , unlike gradient-based models (e.g., neural networks).

•**Log transformation may have distorted important linear relationships** in the data.

### Next Steps

Since MSE was better without scaling , i stuck with using raw features for standard linear regression (as it performed better).

## Project Workflow


1. ### Data Loading & Preprocessing
•Import necessary libraries (NumPy, Pandas, Scikit-learn)

•Load and convert the dataset into a Pandas DataFrame

•Perform exploratory data analysis (EDA)

•Test feature scaling & transformation effects

•Split data into features (X) and target (Y)

2. ### Model Training & Evaluation

•Split the dataset into training (80%) and testing (20%) sets

•Train a linear regression model

•Compute model coefficients to analyze feature importance

•Make predictions on the test set

•Evaluate performance using Mean Squared Error (MSE)

## Results
•**Model Coefficients:**
[ 0.4487,  0.0097, -0.1233,  0.7831, -0.000002, -0.0035, -0.4198, -0.4337 ]

•**Final Mean Squared Error (MSE)** : 0.5559 (using raw features)

## How to Run the Project
1. Clone the repository:

git clone https://github.com/your-repo/california-housing-prediction.git
cd california-housing-prediction


2. Install dependencies:

pip install numpy pandas scikit-learn


3. Run the Python script:

python housing_price_prediction.py



## Future Improvements
• Trying Ridge or Lasso regression, which benefit from feature scaling.

• Feature engineering to improve predictions, such as creating polynomial or interaction terms.

• Exploring non-linear models (e.g., Decision Trees, Random Forests) to capture complex relationships.

## Conclusion

This project demonstrates how linear regression can be used to model housing prices in California.
Interestingly, feature scaling and transformation did not improve performance—suggesting that raw features work well for this dataset in a simple regression model.
