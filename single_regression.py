import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Load data
red_wine = pd.read_csv('data/winequality-red.csv', delimiter=';')
white_wine = pd.read_csv('data/winequality-white.csv', delimiter=';')

# Sample the 'density' data from both red and white wine
sample_size = 100

red_density_raw_sample = red_wine['density'].sample(sample_size, random_state=5310)
white_density_raw_sample = white_wine['density'].sample(sample_size, random_state=5310)

red_quality_sample = red_wine['quality'].sample(sample_size, random_state=5310)[-98:]
white_quality_sample = white_wine['quality'].sample(sample_size, random_state=5310)

# Function to remove outliers based on 1.5 * IQR rule
def remove_outliers(data, label):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    data_no_outliers = data[(data >= lower_bound) & (data <= upper_bound)]

    print(f"Removed {len(data) - len(data_no_outliers)} from {label}.")

    return data_no_outliers

red_density_sample = remove_outliers(red_density_raw_sample, "Red Wine Density")
white_density_sample = remove_outliers(white_density_raw_sample, "White Wine Density")

red_density_sample_normalized = (red_density_sample - red_density_sample.mean()) / red_density_sample.std()
white_density_sample_normalized = (white_density_sample - white_density_sample.mean()) / white_density_sample.std()

#### Reds Wine ####

X = red_density_sample_normalized.reset_index(drop=True)
y = red_quality_sample.reset_index(drop=True)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Predict and calculate residuals
predictions = model.predict(X)
residuals = y - predictions

print(f"Adjusted R^2: {model.rsquared_adj}")

# Plot the data points and line of best fit
plt.figure(figsize=(12, 8))
plt.scatter(red_density_sample_normalized, red_quality_sample, alpha=0.5)
plt.plot(red_density_sample_normalized, predictions, color='red', label="Line of Best Fit")

for i in range(len(y)):
    plt.vlines(red_density_sample_normalized.iloc[i], predictions.iloc[i], y.iloc[i], colors='gray', alpha=0.5)
    
plt.title("Normalized Density vs. Quality with OLS", fontsize=16)
plt.xlabel("Normalized Density", fontsize=16)
plt.ylabel("Quality", fontsize=16)
plt.legend(fontsize=16)
plt.savefig(f'graphs/single_regression/red_osl.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Plot residuals
plt.figure(figsize=(12, 8))
plt.scatter(y, residuals, alpha=0.5, label="Residuals")
plt.axhline(y=0, color='red', linestyle='--', label="Zero Residual Line")
plt.title("Residuals of Linear Regression on Normalized Density vs. Quality", fontsize=16)
plt.xlabel("Observed Quality", fontsize=16)
plt.ylabel("Residuals", fontsize=16)
plt.legend(fontsize=16)
plt.savefig(f'graphs/single_regression/red_residuals.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()

#### White Wine ####

X = white_density_sample_normalized.reset_index(drop=True)
y = white_quality_sample.reset_index(drop=True)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Predict and calculate residuals
predictions = model.predict(X)
residuals = y - predictions

print(f"Adjusted R^2: {model.rsquared_adj}")

# Plot the data points and line of best fit
plt.figure(figsize=(12, 8))
plt.scatter(white_density_sample_normalized, white_quality_sample, alpha=0.5)
plt.plot(white_density_sample_normalized, predictions, color='red', label="Line of Best Fit")

for i in range(len(y)):
    plt.vlines(white_density_sample_normalized.iloc[i], predictions.iloc[i], y.iloc[i], colors='gray', alpha=0.5)
    
plt.title("Normalized Density vs. Quality with OLS", fontsize=16)
plt.xlabel("Normalized Density", fontsize=16)
plt.ylabel("Quality", fontsize=16)
plt.legend(fontsize=16)
plt.savefig(f'graphs/single_regression/white_osl.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Plot residuals
plt.figure(figsize=(12, 8))
plt.scatter(y, residuals, alpha=0.5, label="Residuals")
plt.axhline(y=0, color='red', linestyle='--', label="Zero Residual Line")
plt.title("Residuals of Linear Regression on Normalized Density vs. Quality", fontsize=16)
plt.xlabel("Observed Quality", fontsize=16)
plt.ylabel("Residuals", fontsize=16)
plt.legend(fontsize=16)
plt.savefig(f'graphs/single_regression/white_residuals.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()