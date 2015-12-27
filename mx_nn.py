import sys
sys.path.append('../mxnet/python')
import numpy as np
import pandas as pd
from sklearn import grid_search, metrics, cross_validation, neighbors, pipeline, preprocessing, svm, ensemble, linear_model
import numpy as np
import xgboost as xgb
from scipy import sparse
import mxnet as mx
from mxnet.model import save_checkpoint
import logging
from keras.utils import np_utils

indices = np.fromfile('train.indices', np.int32)
data = np.fromfile('train.data', np.float32)
indptr = np.fromfile('train.indptr', np.int32)
y = np.fromfile('train.y', np.int32)

indices_test = np.fromfile('test.indices', np.int32)
data_test = np.fromfile('test.data', np.float32)
indptr_test = np.fromfile('test.indptr', np.int32)

print (indices.shape, data.shape, indptr.shape, y.shape)
sp = sparse.csr_matrix((data, indices, indptr), dtype=np.float32)
sp_test = sparse.csr_matrix((data_test, indices_test, indptr_test))
print (sp.shape, sp_test.shape)

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(y)

#model_tmp = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1).fit(sp, y)
#indices = np.argsort(model_tmp.feature_importances_)[::-1]
#indices.tofile('indices.tmp')
#qwe
indices = np.fromfile('indices.tmp', dtype=np.int64)
indices = indices[:600]
X = sp[:, indices].todense()
X_test = sp_test[:, indices].todense()

print ('after feature selection: ', X.shape)

#X = np.log(1 + X)
#X_test = np.log(1 + X_test)
pre = preprocessing.StandardScaler()
X = pre.fit_transform(X)
X_test = pre.transform(X_test)

def get_mlp():
    data = mx.symbol.Variable('data')
    x  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=300)
    x = mx.symbol.BatchNorm(data=x)
#    x = mx.symbol.Activation(data = x, name='relu1', act_type="relu")
    x = mx.symbol.LeakyReLU(data=x)
    x = mx.symbol.Dropout(data = x, p=0.5)
    x  = mx.symbol.FullyConnected(data = x, name = 'fc2', num_hidden=300)
    x = mx.symbol.BatchNorm(data=x)
    x = mx.symbol.LeakyReLU(data=x)
#    x = mx.symbol.Activation(data = x, act_type="relu")
    x = mx.symbol.Dropout(data = x, p=0.5)
    
    x  = mx.symbol.FullyConnected(data = x, name = 'fc12', num_hidden=300)
    x = mx.symbol.BatchNorm(data=x)
    x = mx.symbol.LeakyReLU(data=x)    
#    x = mx.symbol.Dropout(data = x, p=0.5)
    
#    x  = mx.symbol.FullyConnected(data = x, name = 'fc22', num_hidden=256)
#    x = mx.symbol.BatchNorm(data=x)
#    x = mx.symbol.LeakyReLU(data=x)
#    x = mx.symbol.Activation(data = x, act_type="relu")
#    x = mx.symbol.Dropout(data = x, p=0.5)
#    x  = mx.symbol.FullyConnected(data = x, name = 'fc3', num_hidden = 200)
#    x = mx.symbol.Activation(data = x, name='relu3', act_type="relu")
#    x = mx.symbol.Dropout(data = x, p=0.5)
    x  = mx.symbol.FullyConnected(data = x, name='fc4', num_hidden=38)
    x  = mx.symbol.SoftmaxOutput(data = x, name = 'softmax')
    return x

def logloss(label, pred_prob):
    label = np_utils.to_categorical(label)
    return metrics.log_loss(label, pred_prob)

res_preds = np.array([])
for i in range(10):
    net = get_mlp()
        
    model = mx.model.FeedForward(
            ctx                = mx.gpu(),
            symbol             = net,
            num_epoch          = 100,
            learning_rate      = 0.01,
            momentum           = 0.9,
            wd                 = 0.000001
#            ,initializer        = mx.init.Xavier(factor_type="in", magnitude=1)
            ,initializer       = mx.initializer.Normal(sigma=0.01)
            )
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size=0.01)
    
    batch_size = 256
    data_shape = (batch_size, 400)
    
    train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size = batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(X_val, y_val, batch_size = batch_size)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    model.fit(X=train_iter, 
              eval_data=val_iter,
              eval_metric=mx.metric.np(logloss),
              batch_end_callback = mx.callback.Speedometer(100, 1000),
    #        epoch_end_callback = do_checkpoint(),
        logger = logger)
    preds = model.predict(X_test)
    res_preds = (res_preds + preds) if res_preds.size else preds
    print (i, res_preds.shape)

res_preds = res_preds / 10

sub = pd.read_csv('sample_submission.csv')
sub.ix[:, 1:] = res_preds
sub.to_csv('res_nn_6.csv', index=False)


  
