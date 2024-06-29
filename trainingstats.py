import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the CSV file
df = pd.read_csv('log/TrainingLog.csv')

# Calculate basic statistics
total_trades = len(df)
winners = df[df['Profit'] > 0]
losers = df[df['Profit'] < 0]
breakeven = df[df['Profit'] == 0]

num_winners = len(winners)
num_losers = len(losers)
num_breakeven = len(breakeven)

win_rate = num_winners / total_trades
loss_rate = num_losers / total_trades
breakeven_rate = num_breakeven / total_trades

mean_profit = df['Profit'].mean()
median_profit = df['Profit'].median()
std_dev_profit = df['Profit'].std()
skewness = df['Profit'].skew()
kurtosis = df['Profit'].kurtosis()

max_profit = df['Profit'].max()
min_profit = df['Profit'].min()

# Print statistics
print(f"Total trades: {total_trades}")
print(f"Number of winners: {num_winners}")
print(f"Number of losers: {num_losers}")
print(f"Number of breakeven trades: {num_breakeven}")
print(f"Win rate: {win_rate:.2%}")
print(f"Loss rate: {loss_rate:.2%}")
print(f"Breakeven rate: {breakeven_rate:.2%}")
print(f"Mean profit: {mean_profit:.4f}")
print(f"Median profit: {median_profit:.4f}")
print(f"Standard deviation of profit: {std_dev_profit:.4f}")
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")
print(f"Max profit: {max_profit:.4f}")
print(f"Min profit: {min_profit:.4f}")

# Create visualizations
plt.figure(figsize=(12, 8))

# Histogram
plt.subplot(2, 2, 1)
sns.histplot(df['Profit'], kde=True)
plt.title('Distribution of Profits')
plt.xlabel('Profit')
plt.ylabel('Frequency')

# Box plot
plt.subplot(2, 2, 2)
sns.boxplot(y=df['Profit'])
plt.title('Box Plot of Profits')
plt.ylabel('Profit')

# Q-Q plot
plt.subplot(2, 2, 3)
stats.probplot(df['Profit'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Profits')

# Cumulative distribution
plt.subplot(2, 2, 4)
sns.ecdfplot(data=df, x='Profit')
plt.title('Cumulative Distribution of Profits')
plt.xlabel('Profit')
plt.ylabel('Cumulative Probability')

plt.tight_layout()
plt.show()

# Additional plot: Profit over time
plt.figure(figsize=(12, 6))
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
plt.plot(df['Date'], df['Profit'].cumsum())
plt.title('Cumulative Profit Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Profit')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import numpy as np

# Calculate log(profit)
# We need to handle negative and zero values, so we'll use np.log1p(x+1) for x >= 0 and -np.log1p(-x+1) for x < 0
df['log_profit'] = np.where(df['Profit'] >= 0, 
                            np.log1p(df['Profit']), 
                            -np.log1p(-df['Profit']))

# Create the plot
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='log_profit', kde=True)
plt.title('Distribution of log(Profit)')
plt.xlabel('log(Profit)')
plt.ylabel('Frequency')

# Add a vertical line at x=0 for reference
plt.axvline(x=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()