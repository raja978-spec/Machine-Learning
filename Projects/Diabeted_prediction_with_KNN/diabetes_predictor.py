# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# df = pd.read_csv('diabetes.csv')

# features = df.drop('Outcome',axis=1)
# labels  = df['Outcome']

# train_feature, test_feature, train_labels, test_labels = train_test_split(features,labels, test_size=0.2)

# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(train_feature, train_labels)

# model_train_prediction = model.predict(train_feature)
# val_score = accuracy_score(train_labels, model_train_prediction)

# model_prediction = model.predict(test_feature)
# test_score = accuracy_score(test_labels, model_prediction)


# # plt.bar(x=['validation_score','test_score'],height=[val_score,test_score])
# # plt.show()

# # Above model gives overfitting result, so that we can change the
# # value of k for n times to get good result on test for that
# # for will be used.

# plt.subplot(4,5)
# for k in range(1,21):
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(train_feature, train_labels)
    
#     model_train_prediction = model.predict(train_feature)
#     val_score = accuracy_score(train_labels, model_train_prediction)

#     model_prediction = model.predict(test_feature)
#     test_score = accuracy_score(test_labels, model_prediction)

#                  EXAMPLE FOR KNN REGRESSION

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv('diabetes.csv')

features = df.drop('Outcome',axis=1)
labels  = df['Outcome']

train_feature, test_feature, train_labels, test_labels = train_test_split(features,labels, test_size=0.2)

model = KNeighborsRegressor()
model.fit(train_feature, train_labels)

param_grid = {
    'n_neighbors':[5,6,7,8],
    'leaf_size':[31,32,33,34]
}

l=len(param_grid['n_neighbors'])*len(param_grid['leaf_size'])

print('Total no of models that are going to build with grid search is',
      l)#16

# This will checks all the combinations from n_neighbors and leaf_size
model_grid_search = GridSearchCV(estimator=model, 
                                 param_grid=param_grid, 
                                 cv=5)
model_grid_search.fit(train_feature, train_labels)
print(model_grid_search.best_params_)
print(model_grid_search.best_score_)

# This will check random values from both hyper parameter
# Here n_iter means that it will checks only 10 combinations
# out of 16 in random way
model_random_search = RandomizedSearchCV(estimator=model,
                                         param_distributions=param_grid,
                                         cv=10,
                                         n_iter=10
                                         )
model_random_search.fit(train_feature, train_labels)
print(model_random_search.best_params_)
print(model_grid_search.best_score_)