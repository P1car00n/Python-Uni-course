from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

data = load_breast_cancer()
df = pd.DataFrame(data.data)

print('Description statistics')
print(df.describe())

plt.title('Box blot with seaborn')
ax = sns.boxenplot(data=df, color='red')
plt.show()

print('Correlation')
corr = df.corr()
print(corr)
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()


plt.title('Heatmap')
sns.heatmap(df)
plt.show()

print('Delete incorrect data')
df.dropna()
print(df)

print('Saving into a file...')
pd.DataFrame.to_csv(df, './out.csv')
