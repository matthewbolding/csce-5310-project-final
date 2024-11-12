import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Load the datasets
red_wine = pd.read_csv('data/winequality-red.csv', delimiter=';')
white_wine = pd.read_csv('data/winequality-white.csv', delimiter=';')

# Define the target variable and model groups for predictors
target = 'quality'
model_1_features = ["density", "residual sugar"]
model_2_features = ["density", "alcohol"]
model_3_features = ["density", "residual sugar", "alcohol"]

# Function to remove outliers based on IQR
def remove_outliers(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        initial_count = len(data)
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        print(f"Removed {initial_count - len(data)} outliers from {column}.")
    return data

# Function to normalize predictors and sample 100 observations
def normalize_and_sample(data, predictors, sample_size=100):
    data = remove_outliers(data, predictors)
    scaler = StandardScaler()
    data[predictors] = scaler.fit_transform(data[predictors])
    data = data.sample(n=sample_size, random_state=5310)
    return data

# Apply outlier removal, normalization, and sampling to each dataset
red_wine = normalize_and_sample(red_wine, model_1_features + model_2_features)
white_wine = normalize_and_sample(white_wine, model_1_features + model_2_features)

# Function to fit a model and print adjusted R-squared
def fit_and_evaluate_model(data, features, target='quality', label=""):
    X = data[features]
    y = data[target]
    X = sm.add_constant(X)  # Adds a constant term to the predictor variables
    model = sm.OLS(y, X).fit()
    print(f"Adjusted R-squared for {label}: {model.rsquared_adj:.4f}")
    return model

# Fit models for the red wine dataset
print("Red Wine Models:")
red_model_1 = fit_and_evaluate_model(red_wine, model_1_features, target, label="Red Wine Model 1: density, residual sugar")
red_model_2 = fit_and_evaluate_model(red_wine, model_2_features, target, label="Red Wine Model 2: density, alcohol")
red_model_3 = fit_and_evaluate_model(red_wine, model_3_features, target, label="Red Wine Model 3: density, residual sugar, alcohol")

# Fit models for the white wine dataset
print("\nWhite Wine Models:")
white_model_1 = fit_and_evaluate_model(white_wine, model_1_features, target, label="White Wine Model 1: density, residual sugar")
white_model_2 = fit_and_evaluate_model(white_wine, model_2_features, target, label="White Wine Model 2: density, alcohol")
white_model_3 = fit_and_evaluate_model(white_wine, model_3_features, target, label="White Wine Model 3: density, residual sugar, alcohol")



