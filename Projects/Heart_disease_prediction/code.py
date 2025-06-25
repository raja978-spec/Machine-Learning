# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import accuracy_score, confusion_matrix
# # from sklearn.compose import ColumnTransformer
# # from sklearn.preprocessing import LabelEncoder
# # import os

# # data = pd.read_csv(os.path.join('Projects','Heart_disease_prediction','data.csv'))
# # #print(data.info())
# # '''
# # <class 'pandas.core.frame.DataFrame'>
# # RangeIndex: 270 entries, 0 to 269
# # Data columns (total 14 columns):
# #  #   Column                   Non-Null Count  Dtype
# # ---  ------                   --------------  -----
# #  0   Age                      270 non-null    int64
# #  1   Sex                      270 non-null    int64
# #  2   Chest pain type          270 non-null    int64
# #  3   BP                       270 non-null    int64
# #  4   Cholesterol              270 non-null    int64
# #  5   FBS over 120             270 non-null    int64
# #  6   EKG results              270 non-null    int64
# #  7   Max HR                   270 non-null    int64
# #  8   Exercise angina          270 non-null    int64
# #  9   ST depression            270 non-null    float64
# #  10  Slope of ST              270 non-null    int64
# #  11  Number of vessels fluro  270 non-null    int64
# #  12  Thallium                 270 non-null    int64
# #  13  Heart Disease            270 non-null    object
# # dtypes: float64(1), int64(12), object(1)
# # memory usage: 29.7+ KB
# # None
# # '''
# # # print(data.isna().sum()) 0 
# # # print(data.isnull().sum()) 0

# # sns.heatmap(data.iloc[:,2:12].corr(), 
# #             annot=True,
# #             fmt='.1f')

# # #plt.show()

# # features = data.iloc[:,2:12]

# # trans = LabelEncoder()
# # labels = trans.fit_transform(data[['Heart Disease']])
# # labels = labels.ravel()
# # print(labels.shape)

# # train_features, test_feature, train_labels, test_labels = train_test_split(features,labels,
# #                                              test_size=0.2)
# # model = LogisticRegression()
# # model.fit(train_features, train_labels)

# # #                 NORMAL HOLD OUT CROSS VALIDATION METHOD
# # '''
# # predicted_lables_for_test = model.predict(test_feature)
# # predicted_lables_for_train = model.predict(train_features)


# # score1 = accuracy_score(test_labels, predicted_lables_for_test)
# # print('test','{:.2f}'.format(score1))

# # con = confusion_matrix(test_labels, predicted_lables_for_test)

# # sns.heatmap(con, annot=True, cbar=True, cmap='Blues')

# # plt.show()

# # score2 = accuracy_score(train_labels, predicted_lables_for_train)
# # print('train','{:.2f}'.format(score2))

# # con1 = confusion_matrix(train_labels, predicted_lables_for_train)

# # sns.heatmap(con1, annot=True, cbar=True, cmap='Blues')
# # plt.show()
# # '''

# # #                     LEAVE ONE OUT CROSS VALIDATION  
# # from sklearn.model_selection import cross_val_score, LeaveOneOut

# # leave_out = LeaveOneOut()

# # cross_score = cross_val_score(model, features, labels, cv=leave_out)
# # print(data.shape)
# # print(cross_score)
# # print(cross_score.shape)
# # print(np.mean(cross_score))
# # '''
# # OUTPUT:
# # (270, 14)
# # [1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
# #  1. 1. 0. 1. 1. 1. 0. 0. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 1. 1. 1. 0.
# #  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 0. 1. 1.
# #  1. 1. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 1.
# #  1. 0. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1.
# #  1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 0. 0.
# #  0. 1. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1.
# #  0. 0. 1. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1. 1. 1. 1.
# #  1. 0. 1. 1. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1.
# #  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 1. 1.
# #  1. 1. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 0. 1.
# #  0. 1. 1. 1. 1. 1.]
# #  (270,)
# # 0.8111111111111111
# # '''






from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
print(type(iris))
# X = iris.data
# y = iris.target
# target_names = iris.target_names

# # PCA transformation
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # t-SNE transformation
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X)

# # Plot PCA
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# for i in range(3):
#     plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_names[i])
# plt.title("PCA Result")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()

# # Plot t-SNE
# plt.subplot(1, 2, 2)
# for i in range(3):
#     plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=target_names[i])
# plt.title("t-SNE Result")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.legend()

# plt.tight_layout()
# plt.show()






