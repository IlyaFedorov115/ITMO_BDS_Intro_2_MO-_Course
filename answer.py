import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('candy-data.csv')

data.head()

data_curr = data.loc[(data['competitorname'] != 'One dime') & (data['competitorname'] != 'Fun Dip') & (data['competitorname'] != 'Milky Way')]
data_curr = data_curr.drop(['winpercent'], axis=1)
data_curr.head()

data_curr['Y']
model = LogisticRegression(random_state = 2019, solver = 'lbfgs').fit(data_curr.drop(['competitorname', 'Y'], axis=1), data_curr[['Y']].values.ravel())

data_test = pd.read_csv('candy-test.csv')
data_test.head(5)

Predict_result = model.predict(data_test.drop(['competitorname', 'Y'], axis=1)[:])
Predict_proba_result = model.predict_proba(data_test.drop(['competitorname', 'Y'], axis=1)[:].values)


predict_df = pd.DataFrame(data={'competitorname': data_test['competitorname'].values,
                                'class': Predict_result, 
                                'proba_0': Predict_proba_result[:,0],
                                'proba_1': Predict_proba_result[:,1]})
predict_df


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

TN, FP, FN, TP = confusion_matrix(data_test['Y'].values, Predict_result).ravel()
AUC = roc_auc_score(data_test['Y'].values, Predict_proba_result[:,1])
print(f'TPR = {TP/(TP+FN)} \n Precision = {TP/(TP+FP)} \n AUC = {AUC}')
















