import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data.csv')

print("Xmean = ", data['X'].mean())
print("Ymean = ", data['Y'].mean())

X_data = np.array(data['X'])
X_data = X_data.reshape((-1, 1))
X_data

Y_data = np.array(data['Y'])
Y_data

model = LinearRegression()
model = model.fit(X_data, Y_data)

r_sq = model.score(X_data, Y_data)
print('coefficient of determination:', r_sq)
