import numpy as np
import pandas as pd
from sklearn import grid_search, metrics, cross_validation, neighbors, pipeline, preprocessing, svm, ensemble, linear_model
import numpy as np
import xgboost as xgb
from scipy import sparse

indices = np.fromfile('train.indices', np.int32)
data = np.fromfile('train.data', np.int32)
indptr = np.fromfile('train.indptr', np.int32)
y = np.fromfile('train.y', np.int32)

indices_test = np.fromfile('test.indices', np.int32)
data_test = np.fromfile('test.data', np.int32)
indptr_test = np.fromfile('test.indptr', np.int32)

print (indices.shape, data.shape, indptr.shape, y.shape)
sp = sparse.csr_matrix((data, indices, indptr))
sp_test = sparse.csr_matrix((data_test, indices_test, indptr_test))
print (sp.shape, sp_test.shape)

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(y)

DX = xgb.DMatrix(sp, label=y)

params = {'booster':'gbtree',
     'max_depth':8,
#     'min_child_weight':4,
     'eta':0.3,
#     'gamma':0.25,
     'silent':1,
     'objective':'multi:softprob',
#     'lambda':1.5,
#     'alpha':1.0,
#      'lambda_bias':0.5,
     'nthread':8,
#      'max_delta_step': 1,
     'subsample':0.9,
     'num_class':38,
      'colsample_bytree':0.9,
     'eval_metric':'mlogloss'
     }
 
xgb.cv(params=params, dtrain=DX, nfold=2, show_progress=True, num_boost_round=100)

#clf = xgb.sklearn.XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=30)
bst = xgb.Booster(params, [DX])
for i in range(80):
    bst.update(DX, i)
    print ('teration: ', i)

#clf.fit(sp, y, eval_metric='mlogloss')
#preds = clf.predict_proba(sp_test)
DT = xgb.DMatrix(sp_test)
preds = bst.predict(DT)

print (preds.shape)
sub = pd.read_csv('sample_submission.csv')
sub.ix[:, 1:] = preds
sub.to_csv('res.csv', index=False, float_format='%.4f')


 