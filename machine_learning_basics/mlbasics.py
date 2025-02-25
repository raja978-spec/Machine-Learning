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

            How simple linear regression works

        Refer simple linear regression.docx

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


  * Polynomial Regression - not all of the model's data fit into linear
                            straight line, some model's data will be 
                            fully fit in slightly curved line,

                            So this is used to fit curved line model's
                            data

                            It is basically used where we can't apply
                            simple and multi linear regression.

                            FORMULA: 
                            REFER: ploynomical.docx for image

 
                            
                HOW POLYNOMIAL REGRESSION IMPLEMENTED

The formula for polynomial regression is 

for degree 2 y = b0 + b1x + b2x**2

for degree 3 y = b0 + b1x + b2x**2 + b3x**3 .... +bnx**n


Sample data set

x_experience = [1,2,3,4,5,6]
y_salary = [367,654,754,865,2444,3344]

consider b is the x_exprience from the formula, we have only b0 now
so we have to find b1,b2,b3, upto bn, that is the done by 


                          EXAMPLE:

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


     * Ridge Regression - It is regularization technique(in regularization
                          we will eliminate some parts of the data to
                          fit the correct data to train model.)

                          It is also like linear regression that finds
                          best fit straight line on data to train model
                          well but the only difference is it adds
                          penalty to train the model with non-overfitted
                          data, that is called L2 regularization.

EX:

Below code has simple dataset which doesn't have overfit, but use this
ridge when train accuracy is greater than test accuracy.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

# Generating some fake data (house sizes and prices)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # House size in 100s of square feet
y = 4 + 3 * X + np.random.randn(100, 1)  # House price with some noise
# print(X[:5])
# print()
# print(y[:5])

# OUTPUT:
# [[0.74908024]
#  [1.90142861]
#  [1.46398788]
#  [1.19731697]
#  [0.31203728]]

# [[6.33428778]
#  [9.40527849]
#  [8.48372443]
#  [5.60438199]
#  [4.71643995]]

train_feature, test_feature, train_label, test_label = train_test_split(X, y, test_size=0.2) 

model = Ridge(alpha=1) # alpha is helps to adjust the penalty added on data
model.fit(train_feature, train_label)

prediction = model.predict(test_feature)
train_prediction = model.predict(train_feature)

train_accuracy = mean_absolute_error(train_label, train_prediction)
t_accuracy = mean_absolute_error(test_label, prediction)
print(f'{t_accuracy} ,{train_accuracy}')

# OUTPUT:
# 0.8740090373056704 ,0.6376585421842711

# Model is will fitted now

* Lasso Regression – Uses L1 regularization to shrink less 
                     important features' coefficients to zero.

* Elastic Net Regression – Combines both L1 (Lasso) and L2 (Ridge) 
                         regularization.

* Logistic Regression – Used for classification, but conceptually 
                        similar to regression.
'''


#                   OVERFITTING, VARIANCE AND REGULARIZATION
'''
 When train accuracy is hight and test accuracy is low overfitting will
 occurred, the are fitted while training, but while we trying to see
 the prediction accuracy on test data there will be a variance
 occur in prediction, this is called the variance.
 This problem can be solved by regularization technique

 In regularization we will eliminate some parts of the data to
 fit the correct data to train model.

 REFER regularization.docx For image


'''
#                          CLASSIFICATION
'''
 Classification in Machine Learning (ML) is a supervised learning 
 technique used to categorize data into predefined labels or 
 classes. The model learns from labeled training data and 
 then predicts the class of new, unseen data.

 TYPES OF CLASSIFICATION

 1. Binary classification - two possible class (spam or not)
                            a  model is trained to classify spam or not
                            spam

 2. Multi-class classification - More than two class (dog, cat, people)
                                 a model is trained to recognize 
                                 in given image either it is dog or cat

                                 It gives only one label at a time

                                 If a image contain both dog or cat
                                 it gives recognize any one from that
                                 image at a time

 3. Multi-label classification - A model trained with this will give you
                                 more than one class at a time,

                                 It will recognize both cat and dog in 
                                 the given image at the same time.


Common Classification Algorithms

Logistic Regression
Decision Trees
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Naïve Bayes
Neural Networks (Deep Learning)

Example Use Cases

Email spam detection (spam vs. not spam)
Sentiment analysis (positive, neutral, negative)
Medical diagnosis (disease classification)
Image recognition (object detection)


'''


#                      K-NEAREST NEIGHBORS
'''
 It is a classification technique that uses set of data points to learn
 to predict new data point.

 ALGORITHM:

 step1: choose one unknow data point
 step2: find the distance between unknow data point to all
        other data point
 step3: if any of the other points are near to the unknow point
        then the unknown point will become a member of that all
        other points.
 step4: predict the response of unknown point.

 For formula and graph image refer  KNN.docx


 EXAMPLE:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv(r'dataset/user_data.csv')
features = dataset.iloc[:,2:4].values
labels = dataset.iloc[:,-1:].values

train_feature, test_feature, train_labels, test_labels = train_test_split(features, labels, test_size=0.25)

sc = StandardScaler() # normalize dataset within specific range
                      # to speed up the training process

#print(train_feature[:5])
train_feature = sc.fit_transform(train_feature)
test_feature = sc.fit_transform(test_feature)
#print(train_feature[:4])

from sklearn.neighbors import KNeighborsClassifier

# metrics is the computation used to calculate the 
# distance between points
# n_neighbors specifies how many neighbor points
# should be found from unknown point.
# p=2 is the value for for euclidean formula(formula 
# to calculate distance, sqrt(other_point, unknown_point**p))
# when we give p=1  d(x, y) = \sum this formula will
# be applied this is called Manhattan Distance (L1 norm)
# which Works well with high-dimensional or sparse data.
#p>2: Less commonly used but gives more weight to larger differences.

classifier = KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')
classifier.fit(train_feature, train_labels)

# We will use confusion metric to see 
# how many correct and wrong prediction
# are made by model. sum of the diagonal elements
# are the correct prediction while others are
# wrong prediction.
from sklearn.metrics import confusion_matrix

model_prediction_labels = classifier.predict(test_feature)

cm = confusion_matrix(test_labels,model_prediction_labels)
print(cm)

OUTPUT: 
[[56  6]
 [ 4 34]]
'''

#                           DECISION TREE
'''
It is one of the supervised learning technique that some feature
from dataset and break down's those feature into different segmentation
to determine the importance of the feature.

Decision tree has several leaf nodes and edge nodes

It will repeat this process for all feature


EX: Consider age is the feature decision tree takes from dataset, it
    will segment like the below one.

                             AGE
                               |
                        |--------------|
                               |
                      young  middle   senior
                              age
                        |
                       Sex
                        |
                      ------
                      |     |
                      F     M

                    
                                ENTROPY

Decision tree uses entropy measurement to make decision tree. This measurement
gives the uncertainty or randomness of the data

Decision tree takes the parameter with high randomness, so that it can
identify all possible value.  

EX: in coin toss head probability - 0 (no randomness) (when a coin has head
                                                       in both side)

                 tail probability - 1 (high randomness) (when a coin head, tail
                                                         both side)


                                                         
            REFER Decision_tree.docx for example dataset with DT pic
                                                         

EX:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

path = r'dataset/salaries.csv'
dataset = pd.read_csv(path)

# labelencoder split feature and lables
# train test split
# 

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
features = dataset.iloc[:,:-1]
labels = dataset.iloc[:,-1:]
df_columns = features.columns

for i in range(len(df_columns)):
    encoded_column = encoder.fit_transform(features[df_columns[i]])
    features[df_columns[i]] = encoded_column

train_feature, test_feature, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=30)

model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(train_feature, train_labels)

prediction = model.predict(test_feature)

score = metrics.accuracy_score(test_labels, prediction)
print(score)


'''


#                     MODEL EVALUATION AND VALIDATION
'''
By evaluating this we can see if the model has overfitted and underfitted.

                           MODEL EVALUATION

Models performance are evaluated using following metrics

1. Accuracy                - sum of correct prediction / total no of prediction
                             
                              Accuracy can be misleading because of
                              imbalanced dataset.

2. Precision and Recall     - Precision measure accuracy of the model's positive prediction.
                              It is mainly used when false positive increases

                              True positive / true positive + false positive 

                              Recall ensures that can a model predicts all
                              true predictions. It is used when False negative
                              is in high.

                              True positive / True positive + False negative

                              In a cancer based system a model doesn't predicts
                              the person who have cancer will make huge risk. 

                              
3. F1 Score   -   Used when both precision and recall is important

                  f1-score = 2 * (precision * recall / precision + recall)

                  It is particularly useful in imbalanced dataset.




                       MODEL  VALIDATION

Used to see how well the model performed on unseen data, or well it 
generalize in unseen data.

1. Train val split - validation set of data are helps to measure how
                     well model predicts unseen data.

                     EX: train_test_split()

                     It helps to prevent over fitting.


2. k- fold cross validation - all the data will become train dataset and
                              validation set. 

                              It splits the data into k fold from that
                              fold some part is used for training and
                              remaining part is used for testing, and finally
                              it calculate mean of all k folds that is
                              the value given by this function.

                              This computation is not good for bigger
                              dataset or complex model.

3. Stratified k-Fold - varation of k-fold that ensure model are trained
                       with all classes.

                       It is particularlly used in classification 
                       imbalanced dataset.


            MONITORING THE MODEL PERFORMANCE DURING TRAINING

It can be done by plotting the loss values of each epoch in training.

EX:

import matplotlib.pyplot as pylt

pylt.plot(train_loss)
pylt.plot(val_loss)
pylt.show()

It is mainly used to detect overfitting and underfitting earlier.

Both train,val loss are decreased in same manner - model learning well
trains loss decreasing, validation loss increasing - overfitting
train loss increase, validation loss increase - underfitting

SOLUTION FOR OVERFITTING:

Accuracy curves, Early stopping(prevent overfitting)

Following are the more solutions to prevent ovefitting

import torch
import torchvision

model = torch.nn.Linear(in_features=10, out_features=5)

#apply L2 regularization
optimizer = torch.optim.SGD(model.parameters, lr=0.01, weight_decay=0.01)
drop_out = torch.nn.Dropout(p=0.5)
data_augmantation = torchvision.transforms.RandomHorizontalFlip()


SOLUTIONS FOR UNDERFITTING

1. Create more neural layers
2. Increase training epoch
3. Reduce regularization
'''


#                  HYPER PARAMETER TUNING
'''
 Process of optimizing hyper paramters like batch_size, learning rate(
 controls the step size of weight update), increase number of layers 
 in model, regularization, increasing no of epoch

'''

#            REPRODUCIBILITY OF MODEL PREDICTION
'''
 It allows other users to verify your model's prediction, it is very
 crucial because various users have various device capability but
 still some of the techniques that can be used on torch to give the
 same good prediction result to all users.

 REFER reproducibility_model.docx for such methods
'''
