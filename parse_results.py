# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:58:29 2017

@author: DeepLearning
"""

import sys
import os
sys.path.insert(0,'Z:/Cristina/Section3/NME_DEC')
sys.path.insert(0,'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels')

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
sns.set(style="whitegrid")

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
#####################################################
## 1) read in the datasets both all NME (to do pretraining)
NME_nxgraphs = r'Z:\Cristina\Section3\breast_MR_NME_biological\NMEs_SER_nxgmetrics'
    
allNMEs_dynamic = pd.read_csv(os.path.join(NME_nxgraphs,'dyn_roi_records_allNMEs_descStats.csv'), index_col=0)
   
allNMEs_morphology = pd.read_csv(os.path.join(NME_nxgraphs,'morpho_roi_records_allNMEs_descStats.csv'), index_col=0)

allNMEs_texture = pd.read_csv(os.path.join(NME_nxgraphs,'text_roi_records_allNMEs_descStats.csv'), index_col=0)

allNMEs_stage1 = pd.read_csv(os.path.join(NME_nxgraphs,'stage1_roi_records_allNMEs_descStats.csv'), index_col=0)

# to load SERw matrices for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_allNMEs_descStats.pklz'), 'rb') as fin:
    nxGdatafeatures = pickle.load(fin)

# to load discrall_dict dict for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGnormfeatures_allNMEs_descStats.pklz'), 'rb') as fin:
    discrall_dict_allNMEs = pickle.load(fin)         

allNME_featurenames = pd.read_csv(os.path.join(NME_nxgraphs,'named_nxGnormfeatures_allNMEs_descStats.csv'), index_col=0)

#########
# shape input (798L, 427L)    
nxGdiscfeatures = discrall_dict_allNMEs   
print('Loading {} all nxGdiscfeatures of size = {}'.format(nxGdiscfeatures.shape[0], nxGdiscfeatures.shape[1]) )

# shape input (798L, 427L)    
combX_allNME = np.concatenate((nxGdiscfeatures, allNMEs_dynamic.as_matrix(), allNMEs_morphology.as_matrix(), allNMEs_texture.as_matrix(), allNMEs_stage1.as_matrix()), axis=1)       
X_allNME_featurenames = np.concatenate((np.vstack(allNME_featurenames.columns),np.vstack(allNMEs_dynamic.columns),np.vstack(allNMEs_morphology.columns),np.vstack(allNMEs_texture.columns),np.vstack(allNMEs_stage1.columns)), axis=0).flatten()  

YnxG_allNME = np.asarray([nxGdatafeatures['roi_id'].values,
        nxGdatafeatures['classNME'].values,
        nxGdatafeatures['nme_dist'].values,
        nxGdatafeatures['nme_int'].values])

print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0])   )

# define variables for DEC 
roi_labels = YnxG_allNME[1]  
roi_labels = ['K' if rl=='U' else rl for rl in roi_labels]

## use y_dec to  minimizing KL divergence for clustering with known classes
ysup = ["{}_{}_{}".format(a, b, c) if b!='nan' else "{}_{}".format(a, c) for a, b, c in zip(YnxG_allNME[1], YnxG_allNME[2], YnxG_allNME[3])]
ysup = ['K'+rl[1::] if rl[0]=='U' else rl for rl in ysup]
classes = [str(c) for c in np.unique(ysup)]
numclasses = [i for i in range(len(classes))]
y_dec = []
for k in range(len(ysup)):
    for j in range(len(classes)):
        if(str(ysup[k])==classes[j]): 
            y_dec.append(numclasses[j])
y_dec = np.asarray(y_dec)

### Read input variables
# shape input (792, 523)     
nxGdiscfeatures = discrall_dict_allNMEs   
print('Loading {} leasions with nxGdiscfeatures of size = {}'.format(nxGdiscfeatures.shape[0], nxGdiscfeatures.shape[1]) )

print('Normalizing dynamic {} leasions with features of size = {}'.format(allNMEs_dynamic.shape[0], allNMEs_dynamic.shape[1]))
normdynamic = (allNMEs_dynamic - allNMEs_dynamic.mean(axis=0)) / allNMEs_dynamic.std(axis=0)
normdynamic.mean(axis=0)
print(np.min(normdynamic, 0))
print(np.max(normdynamic, 0))

print('Normalizing morphology {} leasions with features of size = {}'.format(allNMEs_morphology.shape[0], allNMEs_morphology.shape[1]))
normorpho = (allNMEs_morphology - allNMEs_morphology.mean(axis=0)) / allNMEs_morphology.std(axis=0)
normorpho.mean(axis=0)
print(np.min(normorpho, 0))
print(np.max(normorpho, 0))

print('Normalizing texture {} leasions with features of size = {}'.format(allNMEs_texture.shape[0], allNMEs_texture.shape[1]))
normtext = (allNMEs_texture - allNMEs_texture.mean(axis=0)) / allNMEs_texture.std(axis=0)
normtext.mean(axis=0)
print(np.min(normtext, 0))
print(np.max(normtext, 0))

print('Normalizing stage1 {} leasions with features of size = {}'.format(allNMEs_stage1.shape[0], allNMEs_stage1.shape[1]))
normstage1 = (allNMEs_stage1 - allNMEs_stage1.mean(axis=0)) / allNMEs_stage1.std(axis=0)
normstage1.mean(axis=0)
print(np.min(normstage1, 0))
print(np.max(normstage1, 0))    

# shape input (798L, 427L)    
combX_allNME = np.concatenate((nxGdiscfeatures, normdynamic.as_matrix(), normorpho.as_matrix(), normtext.as_matrix(), normstage1.as_matrix()), axis=1)       
YnxG_allNME = np.asarray([nxGdatafeatures['roi_id'].values,
        nxGdatafeatures['classNME'].values,
        nxGdatafeatures['nme_dist'].values,
        nxGdatafeatures['nme_int'].values])

print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0]) )


'''
####################################################################################################################################
## 6) grid search results of dual-optimization: Z-space DEC LD and num_clusters + MLP classifier
#################################################################################################################################### 
'''  
# set config variables
from decModel_wimgF_dualopt_descStats import *
labeltype = 'wimgF_dualopt_descStats' 
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_wimgF_dualopt_descStats'

# to load a prevously DEC model  
input_size = combX_allNME.shape[1]
latent_size = [input_size/rxf for rxf in [2,5,10]] # other: 10,15,25
varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,10,8)]

scoresM = np.zeros((len(latent_size),len(varying_mu),4))
scoresM_titles=[]

for ic,num_centers in enumerate(varying_mu): 
    for ik,znum in enumerate(latent_size):
        # batch normalization
        X_train = combX_allNME
        y_train = roi_labels

        print('Loading autoencoder of znum = {}, mu = {} , post training DEC results'.format(znum,num_centers))
        dec_model = DECModel(mx.cpu(), X_train, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 

        with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
            dec_model = pickle.load(fu)
          
        with gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
            outdict = pickle.load(fu)
        
        ######## FIND AUC of RF in Z latent space
        datalabels = YnxG_allNME[1]
        X = combX_allNME
        N = X.shape[0]    
        # extact embedding space
        all_iter = mx.io.NDArrayIter({'data': X}, batch_size=X.shape[0], shuffle=False,
                                                  last_batch_handle='pad')   
        ## embedded point zi 
        aDEC = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels')
        # isolate DEC model keys
        dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
            'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
            'encoder_3_bias', 'encoder_2_bias']
        dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
        dec_args['dec_mubestacci'] = dec_model['dec_mubestacci']    
        # convert to list of NDArrays
        mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
        zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X.shape[0], aDEC.xpu).values()[0]      
        # orig paper 256*40 (10240) point for upgrade about 1/6 (N) of data
        #zbestacci = dec_model['zbestacci'] 
        pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
        aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])
        
        zbestacci = dec_model['zbestacci']
        pbestacci = dec_model['pbestacci']

        # compare soft assignments with known labels (only B or M)
        print '\n... MLP fully coneected layer trained on Z_train tested on Z_test' 
        sep = int(zbestacci.shape[0]*0.10)
        #print(zbestacci.shape)
        #print(pbestacci.shape)
        datalabels = np.asarray(datalabels)
        dataZspace = np.concatenate((zbestacci, pbestacci), axis=1) #zbestacci #dec_model['zbestacci']   
        Z = dataZspace[datalabels!='K',:]
        y = datalabels[datalabels!='K']
        #print(Z.shape)

        # Do a 5 fold cross-validation
        Z_test = Z[:sep]
        yZ_test = np.asanyarray(y[:sep]=='M').astype(int) 
        Z_train = Z[sep:]
        yZ_train = np.asanyarray(y[sep:]=='M').astype(int) 
        #print(Z_test.shape)
        #print(Z_train.shape)
        
        # We’ll load MLP using MXNet’s symbolic interface
        dataMLP = mx.sym.Variable('data')
        #The following code declares two fully connected layers with 128 and 32 neurons each. 
        fc1  = mx.sym.FullyConnected(data=dataMLP, num_hidden = 128)
        act1 = mx.sym.Activation(data=fc1, act_type="relu")
        fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 32)
        act2 = mx.sym.Activation(data=fc2, act_type="relu")
        # data has 2 classes
        fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
        # Softmax with cross entropy loss
        mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
        # create a trainable module on CPU     
        batch_size = 50
        mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())

        MLP_train_iter = mx.io.NDArrayIter(Z_train, yZ_train, batch_size, shuffle=True)
        mlp_model.bind(MLP_train_iter.provide_data, MLP_train_iter.provide_label)
        mlp_model.init_params()        
        #        mlp_model.fit(MLP_train_iter,  # train data
        #                          monitor=None,
        #                          optimizer='sgd',  # use SGD to train
        #                          optimizer_params={'learning_rate':0.1},  # use fixed learning rate
        #                          eval_metric= 'acc', #MLPacc(yZ_val, Z_val),  # report accuracy during trainin
        #                          num_epoch=100)
        mlp_model_params = mlp_model.get_params()[0]

        from mxnet import ndarray
        params_dict = ndarray.load(os.path.join(save_to,'mlp_model_params_z{}_mu{}.arg'.format(znum,num_centers)))
        arg_params = {}
        aux_params = {}
        for k, value in params_dict.items():
            arg_type, name = k.split(':', 1)
            if arg_type == 'arg':
                arg_params[name] = value
            elif arg_type == 'aux':
                aux_params[name] = value
            else:
                raise ValueError("Invalid param file ")

        #mlp_model.set_params(arg_params, aux_params)
        #order_params = [(128L, 266L),(128L,),(32L, 128L),(32L,),(2L, 32L),(2L,)]
        # organize weights and biases
        l1=[v.asnumpy().shape for k,v in mlp_model_params.iteritems()]
        k1=[k for k,v in mlp_model_params.iteritems()]
        l2=[v.asnumpy().shape for k,v in arg_params.iteritems()]
        k2=[k for k,v in arg_params.iteritems()]

        for ikparam,sizeparam in enumerate(l1):
            for jkparam,savedparam in enumerate(l2):
                if(sizeparam == savedparam):
                    #print('updating layer parameters: {}'.format(savedparam))
                    mlp_model_params[k1[ikparam]] = arg_params[k2[jkparam]]

        mlp_model.set_params(mlp_model_params, aux_params)
        
        #####################
        # Z-space MLP fully coneected layer for classification
        #####################
        # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=5)
        # Evaluate a score by cross-validation
        tprs = []; aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        cvi = 0
        for train, test in cv.split(Z_train, yZ_train):
            # Multilayer Perceptron
            MLP_val_iter = mx.io.NDArrayIter(Z_train[test], yZ_train[test], batch_size)    
            # prob[i][j] is the probability that the i-th validation contains the j-th output class.
            prob_val = mlp_model.predict(MLP_val_iter)
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(yZ_train[test], prob_val.asnumpy()[:,1])
            # to create an ROC with 100 pts
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            print roc_auc
            aucs.append(roc_auc)
            # plot
            cvi += 1
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        
        ################
        # plot AUC on validation set
        ################
        MLP_heldout_iter = mx.io.NDArrayIter(Z_test, yZ_test, batch_size)   
        probas_heldout = mlp_model.predict(MLP_heldout_iter)
           
        # Compute ROC curve and area the curve
        fpr_val, tpr_val, thresholds_val = roc_curve(yZ_test, probas_heldout.asnumpy()[:, 1])
        auc_val = auc(fpr_val, tpr_val)
        
        ################
        # Calculate NMI: find max soft assignments dec_args
        ################
        N = X_train.shape[0]
        num_classes = len(np.unique(roi_labels)) # present but not needed during AE training
        roi_classes = np.unique(roi_labels)
        y_train_roi_labels = np.asarray(y_train)
        
        W = pbestacci.argmax(axis=1)
        clusters = range(dec_model['num_centers'])
        num_clusters = len(np.unique(W))
        
        MLE_kj = np.zeros((num_clusters,num_classes))
        absWk = np.zeros((num_clusters))
        absCj = np.zeros((num_classes))
        for k in range(num_clusters):
            # find poinst in cluster k
            absWk[k] = np.sum(W==k)
            for j in range(num_classes):
                # find points of class j
                absCj[j] = np.sum(y_train_roi_labels==roi_classes[j])
                # find intersection 
                ptsk = W==k
                MLE_kj[k,j] = np.sum(ptsk[y_train_roi_labels==roi_classes[j]])
        # if not assignment incluster
        absWk[absWk==0]=0.00001
        # compute NMI
        numIwc = np.zeros((num_clusters,num_classes))
        for k in range(num_clusters):
            for j in range(num_classes):
                if(MLE_kj[k,j]!=0):
                    numIwc[k,j] = MLE_kj[k,j]/N * np.log( N*MLE_kj[k,j]/(absWk[k]*absCj[j]) )
                
        Iwk = np.sum(np.sum(numIwc, axis=1), axis=0)       
        Hc = -np.sum(absCj/N*np.log(absCj/N))
        Hw = np.sum(absWk/N*np.log(absWk/N))
        NMI = Iwk/(np.abs(Hc+Hw))
        
        ############# append to matrix
        scoresM[ik,ic,0] = np.mean(aucs)
        scoresM_titles.append("DEC best mean_cvAUC")
        scoresM[ik,ic,1] = std_auc
        scoresM_titles.append("DEC best std_cvAUC")        
        scoresM[ik,ic,2] = auc_val
        scoresM_titles.append("DEC heal-out test AUC")        
        scoresM[ik,ic,3] = NMI
        scoresM_titles.append("DEC train NMI")   


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter

figscoresM, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 12)) 
for k,ax in enumerate(axes.flat):
    im = ax.imshow(scoresM[:,:,k], cmap='viridis', interpolation='nearest')
    ax.grid(False)
    for u in range(len(latent_size)):        
        for v in range(len(varying_mu)):
            ax.text(u,v, '{:.2f}'.format(scoresM[u,v,k]), color=np.array([0.65,0.65,0.65,1]),
                         fontdict={'weight': 'bold', 'size': 10})
    # set ticks
    ax.xaxis.set_major_locator(FixedLocator(np.linspace(0,9,10)))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    
    mu_labels = [str(mu) for mu in varying_mu]
    ax.set_xticklabels(mu_labels, minor=False)
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0,2,3)))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    
    znum_labels = [str(znum) for znum in latent_size]
    ax.set_yticklabels(znum_labels, minor=False)
    ax.xaxis.set_label('Zspace dim')
    ax.yaxis.set_label('# clusters')
    ax.set_title(scoresM_titles[k])

## add colorbar
mins=[]; maxs=[]
# find min and max for scaling
for k in range(3):
    mins.append( np.min(scoresM[:,:,k]) )
    maxs.append( np.max(scoresM[:,:,k]) )
    
minsi = np.min(mins)
maxsi = np.max(maxs)
figscoresM.subplots_adjust(right=0.8)
cbar_ax = figscoresM.add_axes([0.85, 0.15, 0.05, 0.7])
im = ax.imshow(scoresM[:,:,0], cmap='viridis', interpolation='nearest')
figscoresM.colorbar(im, cax=cbar_ax)
#figscoresM.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\scoresM_AUC_{}_final.pdf'.format(labeltype), bbox_inches='tight')    

## plot line plots
dict_aucZlatent = pd.DataFrame() 
for k,znum in enumerate(latent_size):
    for l,num_c in enumerate(varying_mu):
        dict_aucZlatent = dict_aucZlatent.append( pd.Series({'Zspacedim':znum, 
                                                             'Zspace_AUC_ROC': scoresM[k,l,0], 
                                                              'Zspace_test_AUC_ROC': scoresM[k,l,2], 
                                                             'num_clusters':num_c}), ignore_index=True)
    
fig2 = plt.figure(figsize=(20,6))
ax2 = plt.axes()
sns.set_context("notebook")  
sns.pointplot(x="num_clusters", y="Zspace_AUC_ROC", hue="Zspacedim", data=dict_aucZlatent, ax=ax2, size=0.05) 
sns.pointplot(x="num_clusters", y="Zspace_test_AUC_ROC", hue="Zspacedim", data=dict_aucZlatent, ax=ax2, size=0.05, markers=["x","x","x"],linestyles=["--","--","--"]) 
ax2.xaxis.set_label('# clusters')
ax2.yaxis.set_label('Zspace AUC ROC')
ax2.set_title('Zspace AUC ROC vs. number of clusters')
#fig2.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\Zspace AUC ROC_{}.pdf'.format(labeltype), bbox_inches='tight')    
plt.close(fig2)


