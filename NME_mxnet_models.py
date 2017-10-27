# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 18:24:14 2017

@author: windows
"""

"""
Implement linear regression using MXNet APIs.
The function we are trying to learn is: 
y = x1 + 2*x2, where (x1,x2) are input features and y is the corresponding label.
"""


import mxnet as mx
import numpy as np

import logging
logging.getLogger().setLevel(logging.DEBUG)

#Training data
train_data = np.random.uniform(0, 1, [100, 2])
train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in range(100)])
batch_size = 1

#Evaluation Data
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11,26,16])

train_iter = mx.io.NDArrayIter(train_data,train_label, batch_size, shuffle=True,label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)


X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

model = mx.mod.Module(
    symbol = lro ,
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)

mx.viz.plot_network(symbol=lro)

model.fit(train_iter, eval_iter,
            optimizer_params={'learning_rate':0.05, 'momentum': 0.9},
            num_epoch=50,
            eval_metric='mse',
            batch_end_callback = mx.callback.Speedometer(batch_size, 20))
            
## Using a trained model: (Testing and Inference)¶
## Once we have a trained model, we can do a couple of things with it 
## * we can either use it for inference or we can evaluate the trained model on test data. The latter is shown below:
model.predict(eval_iter).asnumpy()
            
## We can also evaluate our model according to some metric. 
## In this example, we are evaluating our model’s mean squared error (MSE) on the evaluation data.
metric = mx.metric.MSE()
model.score(eval_iter, metric)

## Let us try and add some noise to the evaluation data and see how the MSE changes:
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11.1,26.1,16.1]) #Adding 0.1 to each of the values
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)
model.score(eval_iter, metric)


###############################################################################
import random
train_data = np.random.uniform(0, 1, [1000, 10])
train_label = np.array([random.randint(0,1) for n in range(1000)])
batch_size = 1

test_data = train_data[0:100,:]
test_label = train_label[0:100]

############################################################################################
# using iris
import mxnet as mx
import numpy as np

import logging
logging.getLogger().setLevel(logging.DEBUG)

from sklearn import datasets
iris = datasets.load_iris()

#This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica)
# petal and sepal length, stored in a 150x4 numpy.ndarray
#The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.

# Import some data to play with
X = iris.data
y = iris.target
# for a binary classification select only (Setosa, Versicolour)
X, y = X[y != 2], y[y != 2]

# shuffle datasets to reorder 
shuff_ind = np.random.permutation(range(X.shape[0]))
X, y = X[shuff_ind,], y[shuff_ind]

# select 10% of data for testing, 10% for validation, 80% for training
sep = int(X.shape[0]*0.10)
X_test = X[:sep]
y_test = y[:sep]
X_val = X[sep:2*sep]
y_val = y[sep:2*sep]
X_train = X[2*sep:]
y_train = y[2*sep:]
batch_size = 1

train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(X_val, y_val, batch_size)

# Multilayer Perceptron
# We’ll define the MLP using MXNet’s symbolic interface
data = mx.sym.Variable('data')

#MLPs contains several fully connected layers. A fully connected layer or FC layer for short, 
#is one where each neuron in the layer is connected to every neuron in its preceding layer. 
#From a linear algebra perspective, an FC layer applies an affine transform to the n x m input matrix X 
#and outputs a matrix Y of size n x k, where k is the number of neurons in the FC layer. 
#k is also referred to as the hidden size. The output Y is computed according to the equation Y = W X + b. 
#The FC layer has two learnable parameters, the m x k weight matrix W and the m x 1 bias vector b.
#
#In an MLP, the outputs of most FC layers are fed into an activation function, 
#which applies an element-wise non-linearity. This step is critical and it gives 
#neural networks the ability to classify inputs that are not linearly separable. 
#Common choices for activation functions are sigmoid, tanh, and rectified linear unit (ReLU). 
#In this example, we’ll use the ReLU activation function which has several desirable properties
# and is typically considered a default choice.
#
#The following code declares two fully connected layers with 128 and 64 neurons each. 
#Furthermore, these FC layers are sandwiched between ReLU activation layers each 
#one responsible for performing an element-wise ReLU transformation on the FC layer output.
# The first fully-connected layer and the corresponding activation function
fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

# The second fully-connected layer and the corresponding activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")

#The last fully connected layer often has its hidden size equal to the number of
# output classes in the dataset. The activation function for this layer will be 
# the softmax function. 
# The Softmax layer maps its input to a probability score for each class of output. 
# During the training stage, a loss function computes the cross entropy between 
# the probability distribution (softmax output) predicted by the network and the 
# true probability distribution given by the label.

#The output from this layer is fed into a SoftMaxOutput layer that performs 
#softmax and cross-entropy loss computation in one go. Note that loss computation only happens during training.

# data has 2 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
# Softmax with cross entropy loss
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on CPU
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())

mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 20), # output progress for each 100 data batches
              num_epoch=10)  # train for at most 10 dataset passes

#Prediction

#After the above training completes, we can evaluate the trained model by running predictions on test data. 
#The following source code computes the prediction probability scores for each test image. 
# prob[i][j] is the probability that the i-th test image contains the j-th output class.
test_iter = mx.io.NDArrayIter(X_test,  None, batch_size)
prob = mlp_model.predict(test_iter)
print prob.asnumpy()
assert prob.shape == (10, 2)

# Since the dataset also has labels for all test images, we can compute the accuracy metric as follows:
test_iter = mx.io.NDArrayIter(X_test, y_test, batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96


#######################
# using real data
import sys
import os
sys.path.insert(0,'Z:/Cristina/Section3/NME_DEC')

import mxnet as mx
import numpy as np
import pandas as pd

from utilities import *
import data
import model
from autoencoder import AutoEncoderModel
from solver import Solver, Monitor
import logging

from sklearn.manifold import TSNE
from utilities import *
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import sklearn.neighbors 
import matplotlib.patches as mpatches
from sklearn.utils.linear_assignment_ import linear_assignment

try:
   import cPickle as pickle
except:
   import pickle
import gzip

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

#####################################################
from decModel_wimgF_descStats import *

## 1) read in the datasets both all NME (to do pretraining)
NME_imgF = r'Z:\Cristina\Section3\NME_DEC\imgFeatures\NME_nxgraphs'
NME_nxgraphs = r'Z:\Cristina\Section3\breast_MR_NME_biological\NMEs_SER_nxgmetrics'
    
with gzip.open(os.path.join(NME_imgF,'allNMEs_dynamic.pklz'), 'rb') as fin:
    allNMEs_dynamic = pickle.load(fin)

with gzip.open(os.path.join(NME_imgF,'allNMEs_morphology.pklz'), 'rb') as fin:
    allNMEs_morphology = pickle.load(fin)        

with gzip.open(os.path.join(NME_imgF,'allNMEs_texture.pklz'), 'rb') as fin:
    allNMEs_texture = pickle.load(fin)

with gzip.open(os.path.join(NME_imgF,'allNMEs_stage1.pklz'), 'rb') as fin:
    allNMEs_stage1 = pickle.load(fin) 

# to load SERw matrices for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_allNMEs_descStats.pklz'), 'rb') as fin:
    nxGdatafeatures = pickle.load(fin)

# to load discrall_dict dict for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGnormfeatures_allNMEs_descStats.pklz'), 'rb') as fin:
    discrall_dict_allNMEs = pickle.load(fin)           

#########
# shape input (798L, 427L)    
nxGdiscfeatures = discrall_dict_allNMEs   
print('Loading {} all nxGdiscfeatures of size = {}'.format(nxGdiscfeatures.shape[0], nxGdiscfeatures.shape[1]) )

print 'Normalizing dynamic.. '
x_min, x_max = np.min(allNMEs_dynamic, 0), np.max(allNMEs_dynamic, 0)
x_max[x_max==0]=1.0e-07
normdynamic = (allNMEs_dynamic - x_min) / (x_max - x_min)

print 'Normalizing morphology..  '
x_min, x_max = np.min(allNMEs_morphology, 0), np.max(allNMEs_morphology, 0)
x_max[x_max==0]=1.0e-07
normorpho = (allNMEs_morphology - x_min) / (x_max - x_min)

print 'Normalizing texture..  '
x_min, x_max = np.min(allNMEs_texture, 0), np.max(allNMEs_texture, 0)
x_max[x_max==0]=1.0e-07
normtext = (allNMEs_texture - x_min) / (x_max - x_min)

print 'Normalizing stage1.. '
x_min, x_max = np.min(allNMEs_stage1, 0), np.max(allNMEs_stage1, 0)
x_min[np.isnan(x_min)]=1.0e-07
x_max[np.isnan(x_max)]=1.0
normstage1 = (allNMEs_stage1 - x_min) / (x_max - x_min)
normstage1[np.isnan(normstage1)]=1.0e-07

# shape input (798L, 427L)    
combX_allNME = np.concatenate((nxGdiscfeatures, normdynamic, normorpho, normtext, normstage1), axis=1)       
YnxG_allNME = np.asarray([nxGdatafeatures['roi_id'].values,
        nxGdatafeatures['classNME'].values,
        nxGdatafeatures['nme_dist'].values,
        nxGdatafeatures['nme_int'].values])

print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0])   )

# define variables for DEC
labeltype = 'wimgF_descStats' 
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_descStats' #decModel_nagLR001_scheduler'
roi_labels = YnxG_allNME[1]  
roi_labels = ['K' if rl=='U' else rl for rl in roi_labels]

# unbiased validation with held-out set
sep = int(combX_allNME.shape[0]*0.10)
X_val = combX_allNME[:sep]
y_val = roi_labels[:sep]

X_train = combX_allNME[sep:]
y_train = roi_labels[sep:]

# Classification and ROC analysis
datalabels = np.asarray(y_train)
dataspace = X_train
X = dataspace[datalabels!='K',:]
y = np.asarray(datalabels[datalabels!='K']=='M').astype(int)

# to load a prevously DEC model  
znum = 34
num_centers = 3

# Z-space DEC patams optimized with RF classfier by cross-validation
print('Loading autoencoder of znum = {}, mu = {} , post training DEC results'.format(znum,num_centers))
dec_model = DECModel(mx.cpu(), X_train, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 

with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_wimgF_descStats.arg'.format(znum,num_centers)), 'rb') as fu:
    dec_model = pickle.load(fu)

with gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_wimgF_descStats.arg'.format(znum,num_centers)), 'rb') as fu:
    outdict = pickle.load(fu)

# format data
datalabels = np.asarray(y_train)
dataZspace = np.concatenate((outdict['zbestacci'], outdict['pbestacci']), axis=1) #zbestacci #dec_model['zbestacci']   
Z = dataZspace[datalabels!='K',:]
y = np.asarray(datalabels[datalabels!='K']=='M').astype(int)

sep = int(Z.shape[0]*0.10)
Z_test = Z[:sep]
y_test = y[:sep]
Z_val = Z[sep:2*sep]
y_val = y[sep:2*sep]
Z_train = Z[2*sep:]
y_train = y[2*sep:]
batch_size = 10

train_iter = mx.io.NDArrayIter(Z_train, y_train, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(Z_val, y_val, batch_size)

# Multilayer Perceptron
# We’ll define the MLP using MXNet’s symbolic interface
data = mx.sym.Variable('data')

#MLPs contains several fully connected layers. A fully connected layer or FC layer for short, 
fc1  = mx.sym.FullyConnected(data=data, num_hidden=64)
act1 = mx.sym.Activation(data=fc1, act_type="tanh")

# The second fully-connected layer and the corresponding activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 32)
act2 = mx.sym.Activation(data=fc2, act_type="tanh")

#The last fully connected layer often has its hidden size equal to the number of
# data has 2 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
# Softmax with cross entropy loss
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on CPU
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())

mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 20), # output progress for each 100 data batches
              num_epoch=100)  # train for at most 10 dataset passes

#Prediction
#After the above training completes, we can evaluate the trained model by running predictions on test data. 
#The following source code computes the prediction probability scores for each test image. 
# prob[i][j] is the probability that the i-th test image contains the j-th output class.
test_iter = mx.io.NDArrayIter(Z_test,  None, batch_size)
prob = mlp_model.predict(test_iter)
#assert prob.shape == (29, 2)
print(prob.asnumpy())


# Since the dataset also has labels for all test images, we can compute the accuracy metric as follows:
test_iter = mx.io.NDArrayIter(Z_train, y_train, batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
