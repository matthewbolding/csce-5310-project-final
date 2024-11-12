import pandas as pd
from scipy.stats import ttest_ind, t, f
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
red_density_sample = remove_outliers(red_density_raw_sample, "Red wine density")
white_density_sample = remove_outliers(white_density_raw_sample, "White wine density")

red_residual_sugar_sample = remove_outliers(red_residual_sugar_raw_sample, "Red wine residual sugar")
white_residual_sugar_sample = remove_outliers(white_residual_sugar_raw_sample, "White wine residual sugar")

red_alcohol_sample = remove_outliers(red_alcohol_raw_sample, "Red wine alcohol")
white_alcohol_sample = remove_outliers(while_alcohol_raw_sample, "White wine alcohol")

def mean_test_and_ci(sample1, sample2, label, confidence=0.99):
    mean1, mean2 = sample1.mean(), sample2.mean()
    std1, std2 = sample1.std(ddof=1), sample2.std(ddof=1)
    n1, n2 = len(sample1), len(sample2)

    # Perform two-sample t-test
    t_stat, p_value = ttest_ind(sample1, sample2, equal_var=False)
    print(f"Test statistic: {t_stat}; p-value: {p_value}")

    se_diff = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    df = min(n1 - 1, n2 - 1)
    t_crit = t.ppf((1 + confidence) / 2, df)

    margin_of_error = t_crit * se_diff

    diff_mean = mean1 - mean2
    confidence_interval = (diff_mean - margin_of_error, diff_mean + margin_of_error)
    print(f"Confidence Interval: ({confidence_interval[0]}, {confidence_interval[1]})\n")

def sd_test_and_ci(sample1, sample2, label, alpha=0.99):
    var1, var2 = sample1.var(ddof=1), sample2.var(ddof=1)
    n1, n2 = len(sample1), len(sample2)

    # print(f"var1: {var1}; var2: {var2}")

    if var1 > var2:
        f_stat = var1 / var2
        df1 = n1 - 1
        df2 = n2 - 1
    else:
        f_stat = var2 / var1
        df1 = n2 - 1
        df2 = n1 - 1

    # Find the critical F-values for a two-tailed test
    f_critical_low = f.ppf(alpha / 2, df1, df2)
    f_critical_high = f.ppf(1 - alpha / 2, df1, df2)

    # Decision
    print(label)
    print(f"F-stat: {f_stat}; low critical value: {f_critical_low}; high critical value: {f_critical_high}")
    if f_stat < f_critical_low or f_stat > f_critical_high:
        print("Reject the null hypothesis: Variances are significantly different.")
    else:
        print("Fail to reject the null hypothesis: No significant difference in variances.")

    ci_lower = f_stat / f_critical_high
    ci_upper = f_stat * f_critical_low
    print(f"Variance Confidence Interval: ({ci_lower}, {ci_upper})")

def coefficient_of_variation(sample):
    mean = np.mean(sample)
    std_dev = np.std(sample, ddof=1)
    return std_dev / mean if mean != 0 else np.nan

mean_test_and_ci(red_density_sample, red_residual_sugar_sample, "MEAN Red Wine: Density and Residual Sugar")
mean_test_and_ci(red_density_sample, red_alcohol_sample, "MEAN Red Wine: Density and Alcohol")
mean_test_and_ci(white_density_sample, white_residual_sugar_sample, "MEAN White Wine: Density and Residual Sugar")
mean_test_and_ci(white_density_sample, white_alcohol_sample, "MEAN White Wine: Density and Alcohol")

sd_test_and_ci(red_density_sample, red_residual_sugar_sample, "SD Red Wine: Density and Residual Sugar")
print(f"Coefficient of Variation for red_density_sample: {coefficient_of_variation(red_density_sample)}")
print(f"Coefficient of Variation for red_residual_sugar_sample: {coefficient_of_variation(red_residual_sugar_sample)}")
print()

sd_test_and_ci(red_density_sample, red_alcohol_sample, "SD Red Wine: Density and Alcohol")
print(f"Coefficient of Variation for red_density_sample: {coefficient_of_variation(red_density_sample)}")
print(f"Coefficient of Variation for red_alcohol_sample: {coefficient_of_variation(red_alcohol_sample)}")
print()

sd_test_and_ci(white_density_sample, white_residual_sugar_sample, "SD White Wine: Density and Residual Sugar")
print(f"Coefficient of Variation for white_density_sample: {coefficient_of_variation(white_density_sample)}")
print(f"Coefficient of Variation for white_residual_sugar_sample: {coefficient_of_variation(white_residual_sugar_sample)}")
print()


sd_test_and_ci(white_density_sample, white_alcohol_sample, "SD White Wine: Density and Alcohol")
print(f"Coefficient of Variation for white_density_sample: {coefficient_of_variation(white_density_sample)}")
print(f"Coefficient of Variation for red_residual_sugar_sample: {coefficient_of_variation(white_alcohol_sample)}")
print()

