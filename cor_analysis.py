import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# Load data
red_wine = pd.read_csv('data/winequality-red.csv', delimiter=';')
white_wine = pd.read_csv('data/winequality-white.csv', delimiter=';')

# Sample the 'density' data from both red and white wine
sample_size = 100

red_density_raw_sample = red_wine['density'].sample(sample_size, random_state=5310)
white_density_raw_sample = white_wine['density'].sample(sample_size, random_state=5310)

red_residual_sugar_raw_sample = red_wine['residual sugar'].sample(sample_size, random_state=5310)
white_residual_sugar_raw_sample = white_wine['residual sugar'].sample(sample_size, random_state=5310)

red_alcohol_raw_sample = red_wine['alcohol'].sample(sample_size, random_state=5310)
while_alcohol_raw_sample = white_wine['alcohol'].sample(sample_size, random_state=5310)

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

# Remove outliers
red_density_sample = remove_outliers(red_density_raw_sample, "Red wine density")[-90:]
white_density_sample = remove_outliers(white_density_raw_sample, "White wine density")[-90:]

red_residual_sugar_sample = remove_outliers(red_residual_sugar_raw_sample, "Red wine residual sugar")[-90:]
white_residual_sugar_sample = remove_outliers(white_residual_sugar_raw_sample, "White wine residual sugar")[-90:]

red_alcohol_sample = remove_outliers(red_alcohol_raw_sample, "Red wine alcohol")[-90:]
white_alcohol_sample = remove_outliers(while_alcohol_raw_sample, "White wine alcohol")[-90:]

confidence = 1 - 0.05

# Hypothesis tests
red_denisty_sugar = pearsonr(red_density_sample, red_residual_sugar_sample)
print("Red Wine: Density and Sugar")
print(f"Statistic: {red_denisty_sugar.statistic}; p-value: {red_denisty_sugar.pvalue}")
print(red_denisty_sugar.confidence_interval(confidence_level=confidence))
print()

red_denisty_alc = pearsonr(red_density_sample, red_alcohol_sample)
print("Red Wine: Density and Alcohol")
print(f"Statistic: {red_denisty_alc.statistic}; p-value: {red_denisty_alc.pvalue}")
print(red_denisty_alc.confidence_interval(confidence_level=confidence))
print()

white_denisty_sugar = pearsonr(white_density_sample, white_residual_sugar_sample)
print("White Wine: Density and Sugar")
print(f"Statistic: {white_denisty_sugar.statistic}; p-value: {white_denisty_sugar.pvalue}")
print(white_denisty_sugar.confidence_interval(confidence_level=confidence))
print()

white_denisty_alc = pearsonr(white_density_sample, white_alcohol_sample)
print("White Wine: Density and Alcohol")
print(f"Statistic: {white_denisty_alc.statistic}; p-value: {white_denisty_alc.pvalue}")
print(white_denisty_alc.confidence_interval(confidence_level=confidence))