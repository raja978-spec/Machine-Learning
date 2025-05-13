import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

path = os.path.join('ParisHousing.csv')

data = pd.read_csv(path)
# print(data.describe())
# print(data.info())
# print(data.isna().sum())
# print(data.isnull().sum())
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
