#              SUPERVISED LEARNING VS UNSUPERVISED LEARNING
'''

                      SUPERVISED LEARNING

 In supervised learning We feed algorithm with features and labels

 EX: 

 Example: Predicting if a fruit is an Apple or an Orange based on features
python
Copy
Edit
from sklearn.tree import DecisionTreeClassifier

# Step 1: Define the training data (features) and labels
# Features: [weight (grams), texture (0 = Smooth, 1 = Rough)]
X_train = [
    [150, 0],  # Apple
    [130, 0],  # Apple
    [180, 1],  # Orange
    [170, 1],  # Orange
]

# Labels: 0 = Apple, 1 = Orange
y_train = [0, 0, 1, 1]

# Step 2: Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 3: Predict on new data
X_test = [[160, 1]]  # A new fruit with weight 160g and rough texture
prediction = model.predict(X_test)

# Step 4: Output the result
if prediction[0] == 0:
    print("Predicted: Apple 🍏")
else:
    print("Predicted: Orange 🍊")

Explanation:
We assign labels to training data:
Each fruit is labeled as either Apple (0) or Orange (1).
The model learns from labeled data:
It finds patterns in the weight and texture of apples and oranges.
Prediction on new (unlabeled) data:
When we give it a new fruit with weight 160g and rough texture, it predicts whether it's an Apple or an Orange.


                               UNSUPERVISED LEARNING

In unsupervised learning, we do not pass y_train because there are 
no labels. The algorithm tries to find patterns and structure on its own.

For example, in clustering, we only provide input data (X_train), 
and the model groups similar data points without knowing the correct output.

Example: Clustering Fruits Based on Features (Unsupervised Learning)

from sklearn.cluster import KMeans

# Features: [Weight, Texture]
X_train = [
    [150, 0],  # Fruit 1
    [130, 0],  # Fruit 2
    [180, 1],  # Fruit 3
    [170, 1],  # Fruit 4
]

# No labels (Unsupervised Learning)
kmeans = KMeans(n_clusters=2)  # Telling the model to find 2 groups
kmeans.fit(X_train)

# Predict the group of a new fruit
new_fruit = [[160, 1]]  
prediction = kmeans.predict(new_fruit)

print(f"Cluster Assigned: {prediction[0]}")

Key Difference:
Learning Type	Features (X_train)	Labels (y_train)	Example Task
Supervised	✅ Yes	✅ Yes	Classifying fruits as Apple or Orange
Unsupervised	✅ Yes	❌ No	Grouping fruits without knowing their names



                             Why do we use X_train for features and y_train for labels?

                             
In machine learning, it's common to represent:
X_train (Features): The input data (independent variables) used to train the model.
y_train (Labels): The target values (dependent variable) that the model learns to predict.
This notation comes from mathematical conventions where:

X represents input variables (features).
y represents the output variable (label or target).
For example, if we are training a model to predict house prices based on area and number of rooms:

X_train = [[1000, 2], [2000, 3], [1500, 3]]  # Features: [Area, No. of Rooms]
y_train = [150000, 250000, 180000]  # Labels: House prices

The model learns the relationship between X_train and y_train so it can predict 
prices for new houses.
'''

#                       REGRESSION
'''
 Regression is a supervised learning technique used to predict 
 continuous values based on input features. Unlike classification 
 (which predicts discrete categories), regression models estimate 
 numerical outputs.

 Example regression on real world analysis is house price
 prediction. Predicting weather.


                          Types of regression

1. Linear Regression  - Relationship between input and output.

   * Simple linear regression - only single input variable used 
                                to predict output

EX:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# In the below data set we uses area of the house which is 
# only one variable
# to predict house price, so it is considered as
# simple linear regression.
df = pd.read_csv('dataset/homeprices.csv')

plt.title('House price prediction')
plt.xlabel('House Size')
plt.ylabel('Price')
plt.scatter(df[['Area']], df['Price'], marker='+', edgecolors='red')
#plt.show()

model = DecisionTreeRegressor()
model.fit(df[['Area']] , df.Price)

print(df)
size = 3300
price_predict = model.predict([[size]])
print('predicted price for', size, 'is', price_predict)


   * Multiple linear regression - more than one input variables used
                                  REFER multi linear regression.docx for more

   
   EX:

In the below data set we are going to a price with multiple
variables

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('dataset/50_Startups.csv')


x = dataset.iloc[:,:-1] # Taking all columns without profit column
y = dataset.iloc[:,-1].values # profit column values.

# it is easy to predict if all the data in dataset
# are in numerical value, but the column state
# having categorical data that, so we have to do
# transform cat to num using 
# one hot encoding(this means changing the categorical
# data to numeric)
# in below code 'encoder' - is the name of the transformer
# OneHotEncoder - the transform function , [3] index of the column
# where want to do one hot encode, in our case state is in 3rd
# index. remainder will applied on other unspecified columns
# without remainder ct will drops
# all other columns, but in out case we want all columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')

x = np.array(ct.fit_transform(x))

train_feature, test_feature, train_label, test_label = train_test_split(x,y, test_size=0.2)

model = LinearRegression()
model.fit(train_feature, train_label)

predicted_label_price = model.predict(test_feature)

model_prediction_df = pd.DataFrame({'Actual Price': test_label, 
                                    'Predicted Price': 
                                    pd.Series(predicted_label_price)})

                                
             EVALUATE MODEL PERFORMANCE

All these are used in continuos output variable EX: 1,3,8
used in regression

 1. Mean Absolute Error - 1. subtract predicted value with actual value
                          2. sum all the subtracted values / total value - mean
                          3. abs(sum)

 2. Mean Square Error - we squares the means

 3. Root mean squared error 

 EX:

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(test_label, predicted_label_price)))



2. Non linear regression - Non linear combination of input like (curve)


                     How simple linear regression works

        Refer simple linear regression.docx


'''