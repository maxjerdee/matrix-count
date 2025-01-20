# Plot the relationship between error and alpha (indicator for heterogeneity)

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
## TODO: change files
#data = pd.read_csv('results/polbooks_test_margins.csv')
#data = pd.read_csv('results/verbose_test_margins.csv')
data = pd.read_csv('results/more_verbose_test_margins.csv')
#data = pd.read_csv('results/facebook_friends_test_margins.csv')

# Create a scatter plot
plt.figure(figsize=(8, 6))

# Scatter plot for error vs fitted alpha (red color)
plt.scatter(data['fitted_alpha'], data['error'], color='red', label='Error', alpha=0.8, s = 20)

# Scatter plot for error_3 vs fitted alpha (blue color)
plt.scatter(data['fitted_alpha'], data['error_3'], color='blue', label='Error_3', alpha=0.8, s = 20)

print(f"The minimum value of the fitted alpha is: {min(data['fitted_alpha'])}")

# Adding labels and title
plt.xlabel('Fitted Alpha')
plt.ylabel('Error/Error_3')
plt.title('Scatter Plot of Error and Error_3 vs Fitted Alpha')

## TODO: change range for zooming in and zooming out
plt.xlim(0.7, 3)
plt.ylim(0, 0.1)
plt.legend()

# Show the plot
plt.grid(True)
plt.savefig(f'figures/more_verbose_alpha_error.png') ## TODO: change accordingly





# # Filter rows where both estimate and estimate_3 are negative
# filtered_rows = data[(data['estimate'] < 0) | (data['estimate_3'] < 0)]

# # Print m, K, and fitted alpha for the filtered rows
# print(filtered_rows[['m', 'K', 'fitted_alpha']])