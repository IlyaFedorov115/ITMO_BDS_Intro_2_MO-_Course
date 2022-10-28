import pandas as pd
import numpy as np
data = pd.read_csv('candy-data.csv')

data.head(10)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data.corr()
data_test = data.loc[(data['competitorname'] != 'Air Heads') & (data['competitorname'] != 'Kit Kat')]
trg = data_test[['winpercent']]
trn = data_test.drop(['winpercent', 'competitorname', 'Y'], axis=1)

model = LinearRegression()
mod_fit = model.fit(trn, trg)

x_pr = data.loc[data['competitorname'] == 'Air Heads'].drop(['competitorname', 'winpercent', 'Y'], axis=1)
y_pred = model.predict(x_pr)
print('Predict Air Heads: ', y_pred)


x_pr = data.loc[data['competitorname'] == 'Kit Kat'].drop(['competitorname', 'winpercent', 'Y'], axis=1)
y_pred = model.predict(x_pr)
print('Predict Kit Kat: ', y_pred)

x_pr = np.array([1,1,1,0,1,0,1,1,1,0.669,0.456])
x_pr = pd.DataFrame(x_pr.reshape(-1, len(x_pr)), columns=data.drop(['competitorname', 'winpercent', 'Y'], axis=1).columns)
y_pred = model.predict(x_pr)
print('Predict From array: ', y_pred)
