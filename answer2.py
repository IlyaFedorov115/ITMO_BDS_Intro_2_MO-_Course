import pandas as pd
import numpy as np
data = pd.read_csv('data_set.csv',
                       #delimiter=',', 
                       #decimal='.', 
                       index_col = 'id')

from sklearn.neighbors import KNeighborsClassifier

data.head(10)

import matplotlib.pyplot as plt
plt.scatter(data['X'], data['Y'], c=data['Class'],  cmap='winter')
data.corr()


from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(data.drop('Class', axis=1))
scaled_features = scaler.transform(data.drop('Class', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = data.drop('Class', axis=1).columns)
scaled_data

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

scaled_data = data.drop(['Class'], axis=1)
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(scaled_data, data['Class'])


from sklearn.neighbors import NearestNeighbors
scaled_data


neigh = NearestNeighbors(n_neighbors=3, p = 2)
neigh.fit(scaled_data)
print(neigh.kneighbors([[52, 87]]))


model = KNeighborsClassifier(n_neighbors = 3, p =2)
model.fit(scaled_data, data['Class'])
predictions = model.predict([[52, 87]])
predictions

data
