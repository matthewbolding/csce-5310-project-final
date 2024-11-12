import pandas as pd
from scipy.stats import ttest_1samp, t, norm

red_wine = pd.read_csv('data/winequality-red.csv', delimiter=';')
white_wine = pd.read_csv('data/winequality-white.csv', delimiter=';')

red_density = red_wine['density']
white_density = white_wine['density']

sample_size = 100

red_density_raw_sample = red_wine['density'].sample(sample_size, random_state=5310)
white_density_raw_sample = white_wine['density'].sample(sample_size, random_state=5310)

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

# Remove outliers from red and white wine density samples
red_density_sample = remove_outliers(red_density_raw_sample, "Red wine density")
white_density_sample = remove_outliers(white_density_raw_sample, "White wine density")

# Hypothesized mean density
hypothesized_mean = 1.000
confidence_level = 0.99
alpha = 1 - confidence_level

# Perform the one-sample t-test
t_statistic_red, p_value_red = ttest_1samp(red_density_sample, hypothesized_mean)
t_statistic_white, p_value_white = ttest_1samp(white_density_sample, hypothesized_mean)

# Calculate the confidence intervals
# For red wine
red_sample_mean = red_density_sample.mean()
red_sample_std = red_density_sample.std()
red_se = red_sample_std / (len(red_density_sample) ** 0.5)
t_critical = t.ppf(1 - alpha / 2, df=99)  # 99 degrees of freedom for sample size of 100
red_ci_lower = red_sample_mean - t_critical * red_se
red_ci_upper = red_sample_mean + t_critical * red_se

# For white wine
white_sample_mean = white_density_sample.mean()
white_sample_std = white_density_sample.std()
white_se = white_sample_std / (len(white_density_sample) ** 0.5)
white_ci_lower = white_sample_mean - t_critical * white_se
white_ci_upper = white_sample_mean + t_critical * white_se

print(f"Red wine density 99% confidence interval: ({red_ci_lower:.4f}, {red_ci_upper:.4f})")
print(f"White wine density 99% confidence interval: ({white_ci_lower:.4f}, {white_ci_upper:.4f})")

# Interpret the results for red wine
print(f"Red wine test statistic: {t_statistic_red}")
if p_value_red < alpha:
    print(f"Red wine density: Reject the null hypothesis (p-value = {p_value_red:.4f}).")
else:
    print(f"Red wine density: Fail to reject the null hypothesis (p-value = {p_value_red:.4f}).")

# Interpret the results for white wine
print(f"White wine test statistic: {t_statistic_white}")
if p_value_white < alpha:
    print(f"White wine density: Reject the null hypothesis (p-value = {p_value_white:.4f}).")
else:
    print(f"White wine density: Fail to reject the null hypothesis (p-value = {p_value_white:.4f}).")

# Define the threshold
red_threshold = 0.9967
white_threshold = 0.9940

# Calculate observed proportions
red_proportion_below = (red_density_sample < red_threshold).mean()
white_proportion_below = (white_density_sample < white_threshold).mean()

# Hypothesized proportion
hypothesized_proportion = 0.5

# Calculate z-scores for red and white wine
z_red = (red_proportion_below - hypothesized_proportion) / ((hypothesized_proportion * (1 - hypothesized_proportion) / len(red_density_sample)) ** 0.5)
z_white = (white_proportion_below - hypothesized_proportion) / ((hypothesized_proportion * (1 - hypothesized_proportion) / len(white_density_sample)) ** 0.5)

# Calculate two-tailed p-values from the z-scores
p_value_red_prop = 2 * norm.sf(abs(z_red))
p_value_white_prop = 2 * norm.sf(abs(z_white))

# Display results
print(f"Red wine: Proportion below {red_threshold} = {red_proportion_below:.4f}, z-score = {z_red:.4f}, p-value = {p_value_red_prop:.4f}")
print(f"White wine: Proportion below {white_threshold} = {white_proportion_below:.4f}, z-score = {z_white:.4f}, p-value = {p_value_white_prop:.4f}")