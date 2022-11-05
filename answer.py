import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')

data.head(10)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data.drop(['Object', 'Cluster'], axis=1))
scaled_features = scaler.transform(data.drop(['Object', 'Cluster'], axis=1))
scaled_data = pd.DataFrame(scaled_features, columns = data.drop(['Object', 'Cluster'], axis=1).columns)
scaled_data


import matplotlib.pyplot as plt
plt.scatter(scaled_data['X'], scaled_data['Y'], c=data['Cluster'])
plt.scatter(data['X'], data['Y'])

model = KMeans(n_clusters=3, init=np.array([[11.8, 11.6], [8.5, 9.83], [14.0, 14.5]]), max_iter=100, n_init=1)
y_predict = model.fit_predict(data[['X','Y']])
y_predict


model.cluster_centers_
import seaborn as sns


df = data.drop(['Object', 'Cluster'], axis=1)
df['Clusters'] = model.labels_
sns.scatterplot(x="X", y="Y",hue = 'Clusters',  data=df,palette='viridis')

claster_0_cent = model.cluster_centers_[0]
claster_0_cent


Claster_0_count = df[df['Clusters'] == 0]['X'].count()
Claster_0_count


from sklearn.metrics.pairwise import euclidean_distances

sum = 0
#sum = np.sum(np.array())
df[df['Clusters'] == 0][['X','Y']].values
euclidean_distances(df[df['Clusters'] == 0][['X','Y']].values, claster_0_cent.reshape(1, -1)).mean()



