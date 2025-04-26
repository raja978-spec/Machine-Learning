#                SKLEARN SIMPLE IMPUTER
'''
 Helps to fill missing values in dataset.

from sklearn.impute import SimpleImputer
import numpy as np
data=pd.read_csv(path_to_dataset)
si = SimpleImputer(missing_values=np.nan, strategy='median')

numeric_data.iloc[:,:] = si.fit_transform(numeric_data.iloc[:,:])
print(numeric_data.isna().sum())
'''

#      ENCODING CATEGORICAL DATA INTO NUMERIC
'''
One host encoder is the class helps to do this.

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# [1,2,3,6,7,8] columns we want to apply one hot encoder
data=pd.read_csv(path_to_dataset)
trans = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,2,3,6,7,8])],
                          remainder='passthrough')
data = np.array(trans.fit_transform(data))
print(data)

'''

#                      LABEL ENCODER
'''
Helps to encode binary classification values, (Yes, No) values can be
converted into 1,0

import pandas as pd
import numpy as np

data = pd.DataFrame({'x':['yes','no','yes','no']})

from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
data = np.array(en.fit_transform(data)).tolist()
print(data)

OUTPUT:
[1, 0, 1, 0]
'''

#           NORMALIZATION WITH STANDARD SCALER
'''
It uses z-score normalization to put the data in same range.

import pandas as pd
import numpy as np

data = pd.DataFrame({'x':[200,50000,30203,42,13,1,4,51,1,5]})

from sklearn.preprocessing import StandardScaler
nor = StandardScaler()
data['x'] = nor.fit_transform(data)
print(data['x'])

0   -0.472300
1    2.523182
2    1.332388
3   -0.481803
4   -0.483548
5   -0.484270
6   -0.484089
7   -0.481262
8   -0.484270
9   -0.484029
Name: x, dtype: float64
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Given dataset
x_experience = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  #sklearn expectes 2d array
                                                            #that is why shape it transformed
y_salary = np.array([367, 654, 754, 865, 2444, 3344])


poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_experience) # change x into x,x**2, x**3
                                          # which is b0, b1, b2....


model = LinearRegression()
model.fit(x_poly, y_salary) #pass the x_poly

# Predict for smooth curve
x_test = np.linspace(1, 6, 100).reshape(-1, 1)  # Generate smooth X values
x_test_poly = poly.transform(x_test)
y_pred = model.predict(x_test_poly)


plt.figure(figsize=(6,6))

plt.title('Normal plot')
plt.scatter(x_experience, y_salary)

plt.title('Polynomial plot')
plt.plot(x_test, y_pred, color='red')

plt.tight_layout()
plt.show()      
