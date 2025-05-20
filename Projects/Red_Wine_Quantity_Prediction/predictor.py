import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import seaborn as sns
import matplotlib.pyplot as plt
import pathlib as pt

cwd = pt.Path.cwd()
df_path =cwd/'Projects'/'Red_Wine_Quantity_Prediction'/'winequality-red.csv'

df = pd.read_csv(df_path)

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.tight_layout(pad=1.5)
plt.close()
#plt.show()

x = df.drop('quality', axis=1).values  # Shape: (n_samples, n_features)
y = df['quality'].values  # Shape: (n_samples,)

def normalize_data(d):
    normalizer = StandardScaler()
    normalized_data = normalizer.fit_transform(d)
    return normalized_data

X = normalize_data(x)
Y = normalize_data(y.reshape(-1, 1)).ravel()  # Flatten Y to 1D

train_feature, test_feature, train_labels, test_labels = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

train_feature,test_feature,train_lables,test_lables = train_test_split(x,y,
                                                                       test_size=0.2, random_state=42)

lm = Ridge(alpha=1)
lm.fit(train_feature, train_lables)

prevallm = lm.predict(train_feature)
prelm = lm.predict(test_feature)

val_error = mean_absolute_error(train_lables, prevallm)
test_error = mean_absolute_error(test_lables, prelm)

plt.cla()
plt.title('Error Comparision')
plt.xlabel('Errors')
plt.ylabel('Level of errors')
plt.plot(['Validation Error','Test Error'], [val_error,test_error],
         marker='*',mfc='red',mec='gold',ms=20)
plt.tight_layout(pad=10)
#plt.savefig('result2.jpg')
#plt.show()

print('Linear Val error, Test error', val_error, test_error)
# Optional: Try different alphas for Ridge regression
print("\nTuning Ridge Regression Alpha:")
for alpha in [0.01, 0.1, 1, 10, 100]:
    model = Ridge(alpha=alpha)
    model.fit(train_feature, train_labels)
    pred = model.predict(test_feature)
    error = mean_absolute_error(test_labels, pred)
    print(f'Alpha: {alpha:<6} | Test MAE: {error:.4f}')
