# -*- coding: utf-8 -*-
"""
Created on Wed May 03 16:47:40 2017

@author: DeepLearning
"""

##############################################################################################################
# 1) load saved variables
##############################################################################################################
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

'''
##################################################################
## Descriptive datasets for Section3/breast_MR_NME_biological
################################################################## 
'''        
import os
import six.moves.cPickle as pickle
import gzip
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 36})

import seaborn as sns# start by loading nxGdatafeatures
NME_nxgraphs = r'Z:\Cristina\Section3\breast_MR_NME_biological\NMEs_SER_nxgmetrics'

# to load nxGdatafeatures df for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_allNMEs_10binsize.pklz'), 'rb') as fin:
    nxGdatafeatures = pickle.load(fin)

 
## before diving into features summarize labels
print '\n==================== Lesion label:===================='
s = pd.Series(nxGdatafeatures['classNME'], dtype="str")
cat_levels = s.unique()
total = len(s)
for k in range(len(cat_levels)):
    totalcat = np.sum(s == cat_levels[k])
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )

print '\n==================== Lesion diagnosis: ===================='
s = pd.Series(nxGdatafeatures['roi_diagnosis'], dtype="str")
cat_levels = s.unique()
total = len(s)
for k in range(len(cat_levels)):
    totalcat = np.sum(s == cat_levels[k])
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )

print '\n==================== Lesion BIRADS: ===================='
s = pd.Series(nxGdatafeatures['roiBIRADS'], dtype="str")
cat_levels = s.unique()
total = len(s)
for k in range(len(cat_levels)):
    totalcat = np.sum(s == cat_levels[k])
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )


print '\n==================== DCE initial enhancement: ===================='
s = pd.Series(nxGdatafeatures['dce_init'], dtype="str")
cat_levels = s.unique()
total = len(s)
for k in range(len(cat_levels)):
    totalcat = np.sum(s == cat_levels[k])
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )


print '\n==================== DCE delay enhancement: ===================='
s = pd.Series(nxGdatafeatures['dce_delay'], dtype="str")
cat_levels = s.unique()
total = len(s)
for k in range(len(cat_levels)):
    totalcat = np.sum(s == cat_levels[k])
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )

print '\n==================== NME distribution pattern: ===================='
s = pd.Series(nxGdatafeatures['nme_dist'], dtype="str")
cat_levels = s.unique()
total = len(s)
for k in range(len(cat_levels)):
    totalcat = np.sum(s == cat_levels[k])
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )


print '\n==================== NME internal enahcement pattern: ===================='
s = pd.Series(nxGdatafeatures['nme_int'], dtype="str")
cat_levels = s.unique()
total = len(s)
for k in range(len(cat_levels)):
    totalcat = np.sum(s == cat_levels[k]) 
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )    


NME_dist = pd.Series(nxGdatafeatures['nme_dist'], dtype="str")
NME_int_enh = pd.Series(nxGdatafeatures['nme_int'], dtype="str")
label = nxGdatafeatures['classNME'].values
ind = label!='U'
label = label[ind]

# first plot by label
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(221)
ax = sns.countplot(x=NME_dist[ind], order=NME_dist.unique()) # normdiscrallDEL_degree_histogram
ax.set_title('Malignant NME_dist')
ax = fig.add_subplot(222)
ax = sns.countplot(x=NME_int_enh[ind], order=NME_int_enh.unique()) # normdiscrallDEL_degree_histogram
ax.set_title('Malignant NME_int_enh')

ax = fig.add_subplot(223)
ax = sns.countplot(x=NME_dist[~ind], order=NME_dist.unique())
ax.set_title('Benign NME_dist')
ax = fig.add_subplot(224)
ax = sns.countplot(x=NME_int_enh[~ind], order=NME_int_enh.unique())  # normdiscrallDEL_degree_histogram
ax.set_title('Benign NME_int_enh')

fig.set_size_inches(16, 8)
fig.tight_layout()


# gather other infor such as patient levels
import sys
sys.path.insert(0,'Z:\\Cristina\Section3\\breast_MR_NME_biological')
from query_localdatabase import *
from sqlalchemy.orm import sessionmaker, joinedload_all
from sqlalchemy import create_engine
querydb = Querylocaldb()
# configure Session class with desired options
Session = sessionmaker()
queryengine = create_engine('sqlite:///Z:\\Cristina\\Section3\\breast_MR_NME_biological\\nonmass_roibiological.db', echo=False) # now on, when adding new cases # for filled by Z:\\Cristina\\Section3\\NME_DEC\\imgFeatures\\NME_forBIRADSdesc\\nonmass_roirecords.db
Session.configure(bind=queryengine)  # once engine is available
session = Session() #instantiate a Session

list_lesion_ids = nxGdatafeatures['lesion_id'].values
pt_info = pd.DataFrame()
for lesion_id in list_lesion_ids:
    # perform query
    ############# by lesion id
    lesion = session.query(localdatabase.Lesion_record, localdatabase.Radiology_record, localdatabase.ROI_record).\
        filter(localdatabase.Radiology_record.lesion_id==localdatabase.Lesion_record.lesion_id).\
        filter(localdatabase.ROI_record.lesion_id==localdatabase.Lesion_record.lesion_id).\
        filter(localdatabase.Lesion_record.lesion_id == str(lesion_id)).options(joinedload_all('*')).all()
    # print results
    if not lesion:
        print "lesion is empty"
        lesion_id = lesion_id+1
        continue
    
    lesion = lesion[0]    
    # lesion frame       
    casesFrame = pd.Series(lesion.Lesion_record.__dict__)
    pt_info = pt_info.append(casesFrame, ignore_index=True)   


print '\n==================== cad_pt_no_txt: ===================='
cad_pt_no = pd.Series(pt_info['cad_pt_no_txt'])
cad_pt_all = cad_pt_no.unique()
total_patients = len(cad_pt_all)
print 'Number of patients (n=%d) \n' % (total_patients)

print '\n==================== anony_dob_datetime: ===================='
df = pd.DataFrame([])
df['dob_datetimes'] = pt_info['anony_dob_datetime'].toTi
df['exam_datetimes'] = pt_info['exam_dt_datetime']
df['Ageatimaging'] = df['exam_datetimes']-df['dob_datetimes']
df['AgeYear']= df['Ageatimaging'].astype('timedelta64[D]').apply(lambda x: float(x.item())/365.0)
print df['AgeYear'].describe()
session.close_all()
  
'''
##################################################################
## Statistics from two databases of patient collection
################################################################## 
'''  
cad_pt_no = pd.Series(pt_info['cad_pt_no_txt'])
cad_accesion_no = pd.Series(pt_info['exam_a_number_txt'])
ids = pd.concat((cad_pt_no,cad_accesion_no), axis=1)

import sys
sys.path.insert(0,'Z:\\Cristina\\Section3\\NME_DEC\\imgFeatures')

from query_database3 import *
from mylocalbase import db3engine
import database3
from sqlalchemy.orm import sessionmaker, joinedload_all
from sqlalchemy import create_engine

querydb = Querylocaldb3()
Session = sessionmaker()
Session.configure(bind=db3engine)  # once engine is available
session = Session() #instantiate a Session

# gather from db3
pt_infodb3 = pd.DataFrame()
roi_infodb3 = pd.DataFrame()
keepidsnotinbd3 = ids
for j in range(len(ids)):
    cadn,accn = ids.iloc[j]   
    # perform query
    ############# by lesion id
    lesion = session.query(database3.Lesion_record, database3.ROI_record).\
        filter(database3.Lesion_record.lesion_id == database3.ROI_record.lesion_id).\
        filter(database3.Lesion_record.cad_pt_no_txt == str(cadn)).\
        filter(database3.Lesion_record.exam_a_number_txt == str(accn)).all()
    # print results
    if not lesion:
        #print "lesion is empty"
        continue
    else:    
        print cadn,accn
        lesion = lesion[0]
        # frop the lesions since it;s in db2
        keepids_bool = np.logical_and(keepidsnotinbd3['cad_pt_no_txt'] != cadn , keepidsnotinbd3['exam_a_number_txt'] != accn)
        keepidsnotinbd3 = keepidsnotinbd3[keepids_bool]
        
        # lesion frame       
        casesFrame = pd.Series(lesion.Lesion_record.__dict__)
        pt_infodb3 = pt_infodb3.append(casesFrame, ignore_index=True)   
        roiFrame = pd.Series(lesion.ROI_record.__dict__)
        roi_infodb3 = roi_infodb3.append(roiFrame, ignore_index=True)   

session.close_all()

print '\n==================== pt_infodb3: ===================='
cad_pt_no = pd.Series(pt_infodb3['cad_pt_no_txt'])
cad_pt_db3 = cad_pt_no.unique()
total_patientsdb3 = len(cad_pt_db3)
print 'Number of patients db3 (n=%d) \n' % (total_patientsdb3)

print pt_infodb3['lesion_id'].describe()
print pt_infodb3['mri_nonmass_yn'].describe()
  
print '\n==================== roi_label:===================='
s = pd.Series(roi_infodb3['roi_label'], dtype="str")
cat_levels = s.unique()
total = len(s)
for k in range(len(cat_levels)):
    totalcat = np.sum(s == cat_levels[k])
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )



####################################
from query_database2 import *
from mylocalbase import db2engine
import database2
from sqlalchemy.orm import sessionmaker, joinedload_all
from sqlalchemy import create_engine

querydb = Querylocaldb2()
Session = sessionmaker()
Session.configure(bind=db2engine)  # once engine is available
session = Session() #instantiate a Session

# gather from db2
pt_infodb2 = pd.DataFrame()
leftids = keepidsnotinbd3
for j in range(len(keepidsnotinbd3)):
    cadn,accn = keepidsnotinbd3.iloc[j]
    
    # perform query
    ############# by lesion id
    lesion = session.query(database2.Lesion_record, database2.Nonmass_record).\
        filter(database2.Lesion_record.lesion_id == database2.Nonmass_record.lesion_id).\
        filter(database2.Lesion_record.cad_pt_no_txt == str(cadn)).\
        filter(database2.Lesion_record.exam_a_number_txt == str(accn)).all()
    # print results
    if not lesion:
        print "lesion is empty"
        continue
    
    else:    
        print cadn,accn
        # frop the lesions since it;s in db2
        leftids_bool = np.logical_and(keepidsnotinbd3['cad_pt_no_txt'] != cadn , keepidsnotinbd3['exam_a_number_txt'] != accn)
        leftids = leftids[leftids_bool]
        
        lesion = lesion[0]    
        # lesion frame       
        casesFrame = pd.Series(lesion.Lesion_record.__dict__)
        pt_infodb2 = pt_infodb2.append(casesFrame, ignore_index=True)   

session.close_all()

print '\n==================== pt_infodb2: ===================='
cad_pt_no = pd.Series(pt_infodb2['cad_pt_no_txt'])
cad_pt_db2 = cad_pt_no.unique()
total_patientsdb2 = len(cad_pt_db2)
print 'Number of patients db2 (n=%d) \n' % (total_patientsdb2)

print pt_infodb2['lesion_label'].describe()
print pt_infodb2['lesion_id'].describe()
print pt_infodb2['exam_find_mri_nonmass_yn'].describe()

print '\n==================== lesion_label db2:===================='
s = pd.Series(pt_infodb2['lesion_label'], dtype="str")
cat_levels = s.unique()
total = len(s)
for k in range(len(cat_levels)):
    totalcat = np.sum(s == cat_levels[k])
    print '%s (n=%d) (%d/%d = %f percent) \n' % (cat_levels[k], totalcat, totalcat, total, totalcat/float(total) )


'''
#####################################################
## read in the datasets both all NME (to do pretraining)
#####################################################
'''
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
######################
## From Pre-train/fine tune the SAE
######################
'''
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels'
input_size = combX_allNME.shape[1]
latent_size = [input_size/rxf for rxf in [15,10,5,2]]

# train/test splits (test is 10% of labeled data)
sep = int(combX_allNME.shape[0]*0.10)
X_val = combX_allNME[:sep]
y_val = YnxG_allNME[1][:sep]
X_train = combX_allNME[sep:]
y_train = YnxG_allNME[1][sep:]
batch_size = 125 # 160 32*5 = update_interval*5
X_val[np.isnan(X_val)] = 0.00001

allAutoencoders = []
for output_size in latent_size:
    # Train or Read autoencoder: interested in encoding/decoding the input nxg features into LD latent space        
    # optimized for clustering with DEC
    xpu = mx.cpu()
    ae_model = AutoEncoderModel(xpu, [X_train.shape[1],500,500,2000,output_size], pt_dropout=0.2)
    ##  After Pre-train and finetuuning on X_train
    ae_model.load( os.path.join(save_to,'SAE_zsize{}_wimgfeatures_descStats_zeromean.arg'.format(str(output_size))) ) 

    ##  Get train/valid error (for Generalization)
    print "Autoencoder Training error: %f"%ae_model.eval(X_train)
    print "Autoencoder Validation error: %f"%ae_model.eval(X_val)
    # put useful metrics in a dict
    outdict = {'Training set': ae_model.eval(X_train),
               'Testing set': ae_model.eval(X_val),
               'output_size': output_size,
               'sep': sep}

    allAutoencoders.append(outdict)


######################
## Visualize the reconstructed inputs and the encoded representations.
######################
# train/test loss value o
dfSAE_perf = pd.DataFrame()
for SAE_perf in allAutoencoders:
    dfSAE_perf = dfSAE_perf.append( pd.DataFrame({'Reconstruction Error': pd.Series(SAE_perf)[0:2],
                                                  'train/validation':pd.Series(SAE_perf)[0:2].index, 
                                                  'compressed size': SAE_perf['output_size']}) ) 

import seaborn as sns
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(10, 4))
sns.set_color_codes("pastel")
axSAE_perf = sns.pointplot(x="compressed size", y="Reconstruction Error", hue="train/validation", data=dfSAE_perf,  
                           markers=["x","o"], linestyles=["--","-"])  
aa=ax.set_xticklabels(["15x","10x","5x","2x"],fontsize=12)
ax.set_xlabel('compresed size',fontsize=14)
ax.set_ylabel('mean Reconstruccion Loss',fontsize=14)
ax.legend(loc="upper right",fontsize=15)


'''
##################################################################
# Compare formally supervised and semi-supervised learning
## 1) Supervised learning in HD space:
##################################################################
'''
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

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)
RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=0)

# Evaluate a score by cross-validation
sns.set_style("whitegrid")
sns.set_color_codes("dark")
figROCs = plt.figure(figsize=(5,5))    
axaroc = figROCs.add_subplot(1,1,1)              
tprs = []; aucs = []
mean_fpr_OrigX = np.linspace(0, 1, 100)
cvi = 0
for train, test in cv.split(X, y):
    probas = RFmodel.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
    # to create an ROC with 100 pts
    tprs.append(interp(mean_fpr_OrigX, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # plot
    #axaroc = figROCs.add_subplot(1,1,1)
    #axaroc.plot(fpr, tpr, lw=1, alpha=0.5) # with label add: label='cv %d, AUC %0.2f' % (cvi, roc_auc)
    cvi += 1

axaroc.grid(False)
axaroc.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b',alpha=.8)
mean_tpr_OrigX = np.mean(tprs, axis=0)
mean_tpr_OrigX[-1] = 1.0
mean_auc_OrigX = auc(mean_fpr_OrigX, mean_tpr_OrigX)
std_auc_OrigX = np.std(aucs)
axaroc.plot(mean_fpr_OrigX, mean_tpr_OrigX, color='b',
         label=r'cv Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_OrigX, std_auc_OrigX),
         lw=2, alpha=1)     
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr_OrigX + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr_OrigX - std_tpr, 0)
axaroc.fill_between(mean_fpr_OrigX, tprs_lower, tprs_upper, color='grey', alpha=.15,
                 label=r'$\pm$ 1 std. dev.') 

'''
# plot AUC on validation set
'''
y_val_bin = (np.asarray(y_val)=='M').astype(int)
probas_val = RFmodel.fit(X, y).predict_proba(X_val)

# Compute ROC curve and area the curve
fpr_val_OrigX, tpr_val_OrigX, thresholds_val = roc_curve(y_val_bin, probas_val[:, 1])
auc_val_OrigX = auc(fpr_val_OrigX, tpr_val_OrigX)
axaroc.plot(fpr_val_OrigX, tpr_val_OrigX, color='g',linestyle='--',
            label=r'Test (AUC = %0.2f)' % (auc_val_OrigX),
             lw=2, alpha=1)     


axaroc.set_xlabel('False Positive Rate',fontsize=18)
axaroc.set_ylabel('True Positive Rate',fontsize=18)
axaroc.set_title('ROC Original X HD labels={}, all features={} - Supervised cv RF classifier'.format(X.shape[0],X.shape[1]), fontsize=18)
axaroc.legend(loc="lower right",fontsize=18)

'''
##################################################################
# Explore variable importances
##################################################################
'''
ind_featImp = np.argsort(RFmodel.feature_importances_)[::-1]
X_allNME_featurenames[ind_featImp]

# create pandas dF with variable importabces
RFfeatImportances = pd.DataFrame({'varname':X_allNME_featurenames[ind_featImp], 'RFvarImp':np.sort(RFmodel.feature_importances_)[::-1]})

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 32))
# Plot the total crashes
sns.set_color_codes("pastel")
# of all 523 variables seclec those wiht positive varImportances
varDiscrim = RFfeatImportances[RFfeatImportances['RFvarImp']>0.0]

# top 20% = 523*0.20
varDiscrim = RFfeatImportances.iloc[0:int(523*0.20)]
sns.barplot(x="RFvarImp", y="varname", data=varDiscrim, label="varname", color="b")
aa=ax.set_yticklabels(varDiscrim['varname'],fontsize=14)

'''
##################################################################
# 2) Supervised learning in HD space: with only 20% top features
##################################################################
'''
top20_indfea = ind_featImp[0:int(523*0.20)]
top20_combX_allNME = combX_allNME[:,top20_indfea]

# unbiased validation with held-out set
sep = int(top20_combX_allNME.shape[0]*0.10)
X_val = top20_combX_allNME[:sep]
y_val = roi_labels[:sep]

X_train = top20_combX_allNME[sep:]
y_train = roi_labels[sep:]

# Classification and ROC analysis
datalabels = np.asarray(y_train)
dataspace = X_train
X = dataspace[datalabels!='K',:]
y = np.asarray(datalabels[datalabels!='K']=='M').astype(int)

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)
RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=0)

# Evaluate a score by cross-validation
sns.set_style("whitegrid")
sns.set_color_codes("dark")
figROCs = plt.figure(figsize=(4,4))    
axaroc = figROCs.add_subplot(1,1,1)              
tprs = []; aucs = []
mean_fpr_OrigX = np.linspace(0, 1, 100)
cvi = 0
for train, test in cv.split(X, y):
    probas = RFmodel.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
    # to create an ROC with 100 pts
    tprs.append(interp(mean_fpr_OrigX, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # plot
    #axaroc = figROCs.add_subplot(1,1,1)
    #axaroc.plot(fpr, tpr, lw=1, alpha=0.5) # with label add: label='cv %d, AUC %0.2f' % (cvi, roc_auc)
    cvi += 1

axaroc.grid(False)
axaroc.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b',alpha=.8)
mean_tpr_OrigX = np.mean(tprs, axis=0)
mean_tpr_OrigX[-1] = 1.0
mean_auc_OrigX = auc(mean_fpr_OrigX, mean_tpr_OrigX)
std_auc_OrigX = np.std(aucs)
axaroc.plot(mean_fpr_OrigX, mean_tpr_OrigX, color='b',
         label=r'cv Train (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_OrigX, std_auc_OrigX),
         lw=2, alpha=1)     
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr_OrigX + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr_OrigX - std_tpr, 0)
axaroc.fill_between(mean_fpr_OrigX, tprs_lower, tprs_upper, color='grey', alpha=.15,
                 label=r'$\pm$ 1 std. dev.') 

'''
# plot AUC on validation set
'''
y_val_bin = (np.asarray(y_val)=='M').astype(int)
probas_val = RFmodel.fit(X, y).predict_proba(X_val)

# Compute ROC curve and area the curve
fpr_val_OrigX, tpr_val_OrigX, thresholds_val = roc_curve(y_val_bin, probas_val[:, 1])
auc_val_OrigX = auc(fpr_val_OrigX, tpr_val_OrigX)
axaroc.plot(fpr_val_OrigX, tpr_val_OrigX, color='g',linestyle='--',
            label=r'Test (AUC = %0.2f)' % (auc_val_OrigX),
             lw=2, alpha=1)     


axaroc.set_xlabel('False Positive Rate',fontsize=18)
axaroc.set_ylabel('True Positive Rate',fontsize=18)
axaroc.set_title('ROC Original X HD labels={}, all features={} - Supervised cv RF classifier'.format(X.shape[0],X.shape[1]), fontsize=18)
axaroc.legend(loc="lower right",fontsize=18)


'''    
##################################################################
## 2) Unsupervised learning in optimal LD space: Fitting a RF classifier to Zspace
################################################################## 
'''
from decModel_wimgF_dualopt_descStats import *
labeltype = 'decModel_wimgF_AUCopt_descStats' 
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_wimgF_AUCopt_descStats'

# to load a prevously DEC model  
input_size = combX_allNME.shape[1]
latent_size = [input_size/rxf for rxf in [2,5]] # other: 10,15,25
varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,10,8)]

# to load a prevously DEC model  
for znum in latent_size:
    for num_centers in varying_mu: 
        # unbiased validation with held-out set
        sep = int(combX_allNME.shape[0]*0.10)
        X_val = combX_allNME[:sep]
        y_val = roi_labels[:sep]

        X_train = combX_allNME
        y_train = roi_labels
        
        # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=5)
        RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, max_depth=3, random_state=0, verbose=0)

        # Z-space DEC patams optimized with RF classfier by cross-validation
        print('Loading autoencoder of znum = {}, mu = {} , post training DEC results'.format(znum,num_centers))
        dec_model = DECModel(mx.cpu(), X_train, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 

        with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_wimgF_dualopt_descStats.arg'.format(znum,num_centers)), 'rb') as fu:
            dec_model = pickle.load(fu)

        with gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_wimgF_dualopt_descStats.arg'.format(znum,num_centers)), 'rb') as fu:
            outdict = pickle.load(fu)

        # format data
        datalabels = np.asarray(y_train)
        dataZspace = np.concatenate((dec_model['zbestacci'], dec_model['pbestacci']), axis=1) #zbestacci #dec_model['zbestacci']   
        
        # unbiased validation with held-out set
        sep = int(combX_allNME.shape[0]*0.10)
        Z_val = dataZspace[:sep]
        yZ_val = datalabels[:sep]
        
        Z_train = dataZspace[sep:]
        yZ_train = datalabels[sep:]   
        Z = Z_train[yZ_train!='K',:]
        y = np.asarray(yZ_train[yZ_train!='K']=='M').astype(int)
        
        # Evaluate a score by cross-validation
        figROCs = plt.figure(figsize=(9,9))                  
        tprs = []; aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        cvi = 0
        for train, test in cv.split(Z, y):
            probas = RFmodel.fit(Z[train], y[train]).predict_proba(Z[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
            # to create an ROC with 100 pts
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plot
            axaroc = figROCs.add_subplot(1,1,1)
            axaroc.plot(fpr, tpr, lw=1, alpha=0.5) # with label add: label='cv %d, AUC %0.2f' % (cvi, roc_auc)
            cvi += 1

        axaroc.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        axaroc.plot(mean_fpr, mean_tpr, color='b',
                 label=r'MeanROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)     
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axaroc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.') 
        
        ################
        # plot AUC on validation set
        ################
        dec_args_keys = ['encoder_0_bias','encoder_1_bias','encoder_2_bias','encoder_3_bias',
                         'encoder_0_weight','encoder_1_weight','encoder_2_weight','encoder_3_weight']
        dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
        dec_args['dec_mubestacci'] = dec_model['dec_mubestacci'].asnumpy()

        # Evaluate on embedded space
        #####################        
        ## embedded point zi X_val
        all_X_val = mx.io.NDArrayIter({'data': X_val}, 
                                     batch_size=X_val.shape[0], 
                                     shuffle=False, last_batch_handle='pad') 
  
        aDEC = DECModel(mx.cpu(), X_val, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 
        
        mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
        z_val = model.extract_feature(aDEC.feature, mxdec_args, None, all_X_val, X_val.shape[0], aDEC.xpu).values()[0]      
        p_val = np.zeros((z_val.shape[0], dec_model['num_centers']))
        aDEC.dec_op.forward([z_val, dec_args['dec_mubestacci']], [p_val])
    
        # format to predict
        y_val_bin = (np.asarray(y_val)=='M').astype(int)
        Z_val = np.concatenate((z_val, p_val), axis=1)         
        probas_val = RFmodel.fit(Z, y).predict_proba(Z_val)

        # Compute ROC curve and area the curve
        fpr_val, tpr_val, thresholds_val = roc_curve(y_val_bin, probas_val[:, 1])
        auc_val = auc(fpr_val, tpr_val)
        axaroc.plot(fpr_val, tpr_val, color='g',
                    label=r'ValROC (AUC = %0.2f)' % (auc_val),
                     lw=2, alpha=.8)     

        axaroc.set_xlim([-0.05, 1.05])
        axaroc.set_ylim([-0.05, 1.05])
        axaroc.set_xlabel('False Positive Rate',fontsize=18)
        axaroc.set_ylabel('True Positive Rate',fontsize=18)
        axaroc.set_title('ROC LD DEC optimized space={}, all features={} - Unsupervised cv RF classifier'.format(Z.shape[0],Z.shape[1]),fontsize=18)
        axaroc.legend(loc="lower right",fontsize=18)
        plt.show()

'''
##################################################################
# other RF metrics on validation set
##################################################################
'''
pred_X_train_BorM = RFmodel.fit(Z, y).predict(Z)
# compute Z-space Accuracy
Acc_X_train = np.sum(pred_X_train_BorM==y)/float(len(pred_X_train_BorM))
print "cvRF BorM Accuracy in Training = %f " % Acc_X_train

pred_X_val_BorM = RFmodel.fit(Z, y).predict(Z_val)
# compute Z-space Accuracy
Acc_X_val = np.sum(pred_X_val_BorM==y_val_bin)/float(len(pred_X_val_BorM))
print "cvRF BorM Accuracy in Validation = %f " % Acc_X_val



'''
##################################################################
## 3) Unsupervised learning in optimal LD space: DEC + fully connected MLP classifier
##################################################################
'''
# set config variables
from decModel_wimgF_dualopt_descStats import *
labeltype = 'decModel_wimgF_AUCopt_descStats' 
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_wimgF_AUCopt_descStats'

# to load a prevously DEC model  
input_size = combX_allNME.shape[1]
latent_size = [input_size/rxf for rxf in [15,10,5,2]] # other: 10,15,25
varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,10,8)]

# to load a prevously DEC model  
for znum in latent_size:
    cvRForigXAUC = []
    initAUC = []
    valAUC = []
    cvRFZspaceAUC = []           
    normalizedMI = []
    for num_centers in varying_mu: 
        # batch normalization
        X_train = combX_allNME
        y_dec_train = y_dec
        y_train = roi_labels

        print('Loading autoencoder of znum = {}, mu = {} , post training DEC results'.format(znum,num_centers))
        dec_model = DECModel(mx.cpu(), X_train, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 

        with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
            dec_model = pickle.load(fu)

        with gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
            outdict = pickle.load(fu)

        print('DEC train init AUC = {}'.format(outdict['meanAuc_cv'][0]))
        max_meanAuc_cv = max(outdict['meanAuc_cv'])
        indmax_meanAuc_cv = outdict['meanAuc_cv'].index(max_meanAuc_cv)
        print('DEC train max AUC = {}'.format(max_meanAuc_cv))
        print('DEC validation AUC at max train = {}'.format(outdict['auc_val'][indmax_meanAuc_cv]))
        
        max_auc_val = max(outdict['auc_val'])
        indmax_auc_val = outdict['auc_val'].index(max_auc_val)
        print('DEC train AUC at max val = {}'.format(outdict['meanAuc_cv'][indmax_auc_val]))
        print('DEC validation AUC at max val = {}'.format(max_auc_val))
        print('DEC resulting NMI={}'.format(outdict['NMI']))
        
        # to plot metrics for all varying_mu
        cvRForigXAUC.append(0.67)
        initAUC.append(outdict['meanAuc_cv'][0])
        cvRFZspaceAUC.append(outdict['meanAuc_cv'][indmax_meanAuc_cv])
        valAUC.append(outdict['auc_val'][indmax_meanAuc_cv])
        # or to calculate NMI
        # save output results
        dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
        'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
        'encoder_3_bias', 'encoder_2_bias']
        dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
        dec_args['dec_mubestacci'] = dec_model['dec_mubestacci']
        
        # Calculate normalized MI: find the relative frequency of points in Wk and Cj
        #####################
        N = X_train.shape[0]
        num_classes = len(np.unique(roi_labels)) # present but not needed during AE training
        roi_classes = np.unique(roi_labels)
        y_train_roi_labels = np.asarray(y_train)
        
        # extact embedding space
        all_iter = mx.io.NDArrayIter({'data': X_train}, batch_size=X_train.shape[0], shuffle=False,
                                                  last_batch_handle='pad')   
        ## embedded point zi 
        aDEC = DECModel(mx.cpu(), X_train, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 
        mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
        
        # gather best-Zspace or dec_model['zbestacci']
        zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X_train.shape[0], aDEC.xpu).values()[0]      
        # compute model-based best-pbestacci or dec_model['pbestacci']
        pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
        aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])
        
        # find max soft assignments dec_args
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
        print("... num_centers={} DEC normalizedMI = {}".format(num_centers,NMI))
        normalizedMI.append(NMI)
        print("========================\n")
    
    # plot latent space Accuracies vs. original
    colors = plt.cm.jet(np.linspace(0, 1, 16))
    fig2 = plt.figure(figsize=(20,6))
    #ax2 = plt.axes()
    sns.set_context("notebook")
    ax1 = fig2.add_subplot(2,1,1)
    ax1.plot(varying_mu, cvRFZspaceAUC, color=colors[0], ls=':', label='cvRFZspaceAUC')
    ax1.plot(varying_mu, valAUC, color=colors[2], label='valAUC')
    ax1.plot(varying_mu, initAUC, color=colors[4], label='initAUC')
    ax1.plot(varying_mu, cvRForigXAUC, color=colors[6], label='cvRForigXAUC')
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':12})
    
    ax2 = fig2.add_subplot(2,1,2)
    ax2.plot(varying_mu, normalizedMI, color=colors[8], label='normalizedMI')
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h2, l2, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':12})
    ax2.set_xlabel('# clusters',fontsize=14)
    ax1.set_title('Performance of DEC optimized Z-space classification, Reduction of HD size by x{}, LD size={}'.format(input_size/znum, znum),fontsize=14)
        
            
        

'''
##################################################################
## 4) Unsupervised learning in optimal LD space: Fitting aN MLP DUAL OPTIMIZATION
################################################################## 
'''
import logging
logging.getLogger().setLevel(logging.INFO)

# set config variables
from decModel_wimgF_dualopt_descStats import *
labeltype = 'wimgF_dualopt_descStats_saveparams' 
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_wimgF_dualopt_descStats_saveparams'

# to load a prevously DEC model  
input_size = combX_allNME.shape[1]
latent_size = [input_size/rxf for rxf in [15,10,5,2]]
varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,10,8)]

scoresM = np.zeros((len(latent_size),len(varying_mu),4))
scoresM_titles=[]

######################
# DEC: define num_centers according to clustering variable
######################   
# to load a prevously DEC model  
for ik,znum in enumerate(latent_size):
    cvRForigXAUC = []
    initAUC = []
    valAUC = []
    cvRFZspaceAUC = [] 
    normalizedMI = []
    for ic,num_centers in enumerate(varying_mu): 
        X = combX_allNME
        y = roi_labels
        y_train_roi_labels = np.asarray(y)

        print('Loading autoencoder of znum = {}, mu = {} , post training DEC results'.format(znum,num_centers))
        dec_model = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 

        with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
            dec_model = pickle.load(fu)
          
        with gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fu:
            outdict = pickle.load(fu)
        
        print('DEC train init AUC = {}'.format(outdict['meanAuc_cv'][0]))
        max_meanAuc_cv = outdict['meanAuc_cv'][-1]
        indmax_meanAuc_cv = outdict['meanAuc_cv'].index(max_meanAuc_cv)
        print r'DEC train max meanAuc_cv = {} $\pm$ {}'.format(max_meanAuc_cv,dec_model['std_auc'][indmax_meanAuc_cv])
        print('DEC validation AUC at max meanAuc_cv = {}'.format(outdict['auc_val'][indmax_meanAuc_cv]))
        
        #####################
        # extract Z-space from optimal DEC model
        #####################
        # saved output results
        dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
        'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
        'encoder_3_bias', 'encoder_2_bias']
        dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
        dec_args['dec_mubestacci'] = dec_model['dec_mu']
        
        N = X.shape[0]
        all_iter = mx.io.NDArrayIter({'data': X}, batch_size=X.shape[0], shuffle=False,
                                                  last_batch_handle='pad')   
        ## extract embedded point zi 
        mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
        aDEC = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 
        
        # gather best-Zspace or dec_model['zbestacci']
        zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X.shape[0], aDEC.xpu).values()[0]      
        zbestacci = dec_model['zbestacci']
        
        # compute model-based best-pbestacci or dec_model['pbestacci']
        pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
        aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])
        pbestacci = dec_model['pbestacci']
        
        # pool Z-space variables
        datalabels = np.asarray(y)
        dataZspace = np.concatenate((zbestacci, pbestacci), axis=1) 

        #####################
        # unbiased assessment: SPlit train/held-out test
        #####################
        # to compare performance need to discard unkown labels, only use known labels (only B or M)
        Z = dataZspace[datalabels!='K',:]
        y = datalabels[datalabels!='K']
      
        print '\n... MLP fully coneected layer trained on Z_train tested on Z_test' 
        sep = int(X.shape[0]*0.10)
        Z_test = Z[:sep]
        yZ_test = np.asanyarray(y[:sep]=='M').astype(int) 
        Z_train = Z[sep:]
        yZ_train = np.asanyarray(y[sep:]=='M').astype(int) 
       
        # We’ll load MLP using MXNet’s symbolic interface
        dataMLP = mx.sym.Variable('data')
        # MLP: two fully connected layers with 128 and 32 neurons each. 
        fc1  = mx.sym.FullyConnected(data=dataMLP, num_hidden = 128)
        act1 = mx.sym.Activation(data=fc1, act_type="relu")
        fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 32)
        act2 = mx.sym.Activation(data=fc2, act_type="relu")
        # data has 2 classes
        fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
        # Softmax output layer
        mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
        # create a trainable module on CPU     
        batch_size = 50
        mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
        # pass train/test data to allocate model (bind state)
        MLP_train_iter = mx.io.NDArrayIter(Z_train, yZ_train, batch_size, shuffle=False)
        mlp_model.bind(MLP_train_iter.provide_data, MLP_train_iter.provide_label)
        mlp_model.init_params()   
        mlp_model.init_optimizer()
        mlp_model_params = mlp_model.get_params()[0]
        
        # update parameters based on optimal found during cv Training
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

        # order of params: [(128L, 266L),(128L,),(32L, 128L),(32L,),(2L, 32L),(2L,)]
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
        # upddate model parameters
        mlp_model.set_params(mlp_model_params, aux_params)
        
        #####################
        # ROC: Z-space MLP fully coneected layer for classification
        #####################
        figROCs = plt.figure(figsize=(9,9))    
        axaroc = figROCs.add_subplot(1,1,1)
        # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=5)
        # Evaluate a score by cross-validation
        tprs = []; aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        cvi = 0
        for train, test in cv.split(Z_train, yZ_train):
            # Pass cv data
            MLP_val_iter = mx.io.NDArrayIter(Z_train[test], None, batch_size)    
            # prob[i][j] is the probability that the i-th validation contains the j-th output class.
            prob_val = mlp_model.predict(MLP_val_iter)
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(yZ_train[test], prob_val.asnumpy()[:,1])
            # to create an ROC with 100 pts
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plot
            #axaroc.plot(fpr, tpr, lw=1, alpha=0.6) # with label add: label='cv %d, AUC %0.2f' % (cvi, roc_auc)
            cvi += 1
            
        axaroc.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        axaroc.plot(mean_fpr, mean_tpr, color='b',label=r'MeanROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=3, alpha=1)     
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axaroc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1,label=r'$\pm$ 1 std. dev.') 
        
        ################
        # plot AUC on validation set
        ################
        MLP_heldout_iter = mx.io.NDArrayIter(Z_test, None, batch_size)   
        probas_heldout = mlp_model.predict(MLP_heldout_iter)
           
        # Compute ROC curve and area the curve
        fpr_val, tpr_val, thresholds_val = roc_curve(yZ_test, probas_heldout.asnumpy()[:, 1])
        auc_val = auc(fpr_val, tpr_val)
        axaroc.plot(fpr_val, tpr_val, color='r',label=r'ValROC (AUC = %0.2f)' % (auc_val),lw=3, alpha=1)     

        axaroc.set_xlim([-0.05, 1.05])
        axaroc.set_ylim([-0.05, 1.05])
        axaroc.set_xlabel('False Positive Rate',fontsize=18)
        axaroc.set_ylabel('True Positive Rate',fontsize=18)
        axaroc.set_title('ROC LD DEC optimized space={}, all features={} - Unsupervised DEC + cv MLP classifier'.format(Z.shape[0],Z.shape[1]),fontsize=18)
        axaroc.legend(loc="lower right",fontsize=18)
        plt.show()
        
        cvRForigXAUC.append(0.67)
        initAUC.append(dec_model['meanAuc_cv'][0])
        cvRFZspaceAUC.append(np.mean(aucs))
        valAUC.append(auc_val)
        
        ################
        # Calculate NMI: find max soft assignments dec_args
        ################
        num_classes = len(np.unique(roi_labels))
        roi_classes = np.unique(roi_labels)
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
        Hc = np.sum(absCj/N*np.log(absCj)/N)
        Hw = np.sum(absWk/N*np.log(absWk)/N)
        NMI = Iwk/(np.abs(Hc+Hw)/2)
        print("... num_centers={} DEC normalizedMI = {}".format(num_centers,NMI))
        
        
        numclassesNMI = [i for i in range(len(roi_classes))]
        y_NMI = []
        for k in range(len(y_train_roi_labels)):
            for j in range(len(roi_classes)):
                if(str(y_train_roi_labels[k])==roi_classes[j]): 
                    y_NMI.append(numclassesNMI[j])
        y_NMI = np.asarray(y_NMI)
        from sklearn.metrics.cluster import normalized_mutual_info_score
        NMI_info_score = normalized_mutual_info_score(W,y_NMI)
        
        print("... num_centers={} DEC normalized_mutual_info_score = {}".format(num_centers,NMI_info_score))
        normalizedMI.append(NMI_info_score)
        print("========================\n")
        
        ############# append to matrix
        scoresM[ik,ic,0] = np.mean(aucs)
        scoresM_titles.append("DEC best mean_cvAUC")
        scoresM[ik,ic,1] = std_auc
        scoresM_titles.append("DEC best std_cvAUC")        
        scoresM[ik,ic,2] = auc_val
        scoresM_titles.append("DEC heal-out test AUC")        
        scoresM[ik,ic,3] = NMI
        scoresM_titles.append("DEC train NMI")   
    
    # plot latent space Accuracies vs. original
    colors = plt.cm.jet(np.linspace(0, 1, 16))
    fig2 = plt.figure(figsize=(12,6))
    #ax2 = plt.axes()
    sns.set_context("notebook")
    ax1 = fig2.add_subplot(2,1,1)
    ax1.plot(varying_mu, cvRFZspaceAUC, color=colors[0], ls=':', label='cvRFZspaceAUC')
    ax1.plot(varying_mu, valAUC, color=colors[2], label='valAUC')
    ax1.plot(varying_mu, initAUC, color=colors[4], label='initAUC')
    ax1.plot(varying_mu, cvRForigXAUC, color=colors[6], label='cvRForigXAUC')
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':16})
    
    ax2 = fig2.add_subplot(2,1,2)
    ax2.plot(varying_mu, normalizedMI, color=colors[8], label='normalizedMI')
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h2, l2, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':16})
    ax2.set_xlabel('# clusters',fontsize=16)
    ax1.set_title('Performance of DEC optimized Z-space classification, Reduction of HD size by x{}, LD size={}'.format(input_size/znum, znum),fontsize=20)
          
           
'''
##################################################################
## 5) Find best performing parameters
################################################################## 
'''        

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter
import matplotlib.cm

figscoresM, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18)) 

for k,ax in enumerate(axes.flat):
    im = ax.imshow(scoresM[:,:,k], cmap=matplotlib.cm.get_cmap('Blues'), interpolation='nearest')
    ax.grid(False)
    for u in range(len(latent_size)):        
        for v in range(len(varying_mu)):
            ax.text(v,u,'{:.3f}'.format(scoresM[u,v,k]), color=np.array([0.15,0.15,0.15,0.9]),
                         fontdict={'weight': 'bold', 'size': 16})
    # set ticks
    ax.xaxis.set_major_locator(FixedLocator(np.linspace(0,9,10)))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    
    mu_labels = [str(mu) for mu in varying_mu]
    ax.set_xticklabels(mu_labels, minor=False,fontsize=16)
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0,3,4)))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    
    znum_labels = ['15x','10x','5x','2x'] #[str(znum) for znum in latent_size]
    ax.set_yticklabels(znum_labels, minor=False,fontsize=16)
    ax.xaxis.set_label('latent space reduction')
    ax.yaxis.set_label('# cluster centroids')
    ax.set_title(scoresM_titles[k],fontsize=16)
    
  
## plot line plots
dict_aucZlatent = pd.DataFrame() 
for k,znum in enumerate(latent_size):
    for l,num_c in enumerate(varying_mu):
        dict_aucZlatent = dict_aucZlatent.append( pd.Series({'Zspacedim':znum, 
                                                             'Zspace_AUC_ROC': scoresM[k,l,0], 
                                                              'Zspace_test_AUC_ROC': scoresM[k,l,2], 
                                                             'num_clusters':num_c}), ignore_index=True)

fig2 = plt.figure(figsize=(20,10))
ax2 = plt.axes()
sns.set_context("notebook")  
sns.pointplot(x="num_clusters", y="Zspace_AUC_ROC", hue="Zspacedim", data=dict_aucZlatent, ax=ax2, size=0.05) 
sns.pointplot(x="num_clusters", y="Zspace_test_AUC_ROC", hue="Zspacedim", data=dict_aucZlatent, ax=ax2, size=0.05, markers=["x","x","x","x"],linestyles=["--","--","--","--"]) 
ax2.xaxis.set_label('# clusters')
ax2.yaxis.set_label('Zspace AUC ROC')
ax2.set_title('Zspace AUC ROC vs. number of clusters',fontsize=20)

# plot box plots
fig = plt.figure(figsize=(12,18))
dict_aucZlatent['Zspacedim_cats'] = pd.Series(dict_aucZlatent['Zspacedim'], dtype="category")
ax1 = fig.add_subplot(2,2,1)
sns.boxplot(y="Zspace_AUC_ROC", x="Zspacedim_cats", data=dict_aucZlatent, ax=ax1)
znum_labels = ['15x','10x','5x','2x'] 
ax1.set_xticklabels(znum_labels, minor=False,fontsize=12)
ax1.set_xlabel('Latent space reduction')
ax1.set_ylabel('cv Train AUC ROC across # of centroids')

ax2 = fig.add_subplot(2,2,3)
dict_aucZlatent['num_clusters_cats'] = pd.Series(dict_aucZlatent['num_clusters'], dtype="category")
sns.boxplot(y="Zspace_AUC_ROC", x="num_clusters_cats", data=dict_aucZlatent, ax=ax2)
ax2.set_xlabel('# cluster centroids')
ax2.set_ylabel('cv Train AUC ROC across reduction ratios')

ax3 = fig.add_subplot(2,2,2)
sns.boxplot(y="Zspace_test_AUC_ROC", x="Zspacedim_cats", data=dict_aucZlatent, ax=ax3)
ax3.set_xticklabels(znum_labels, minor=False,fontsize=12)
ax3.set_xlabel('Latent space reduction')
ax3.set_ylabel('Test AUC ROC across # of centroids')

ax4 = fig.add_subplot(2,2,4)
sns.boxplot(y="Zspace_test_AUC_ROC", x="num_clusters_cats", data=dict_aucZlatent, ax=ax4)
ax4.set_xlabel('# cluster centroids')
ax4.set_ylabel('Test AUC ROC across reduction ratios')


    


# find best performing by the average of broth train and test performance
max_aucZlatent = np.max(dict_aucZlatent["Zspace_AUC_ROC"])
indmax_meanAuc_cv = dict_aucZlatent["Zspace_AUC_ROC"]== max_aucZlatent
print "\n================== Best average train/test perfomance parameters:" 
bestperf_params = dict_aucZlatent[indmax_meanAuc_cv]
print bestperf_params


Zspacedim_best = int(bestperf_params.iloc[0]['Zspacedim'])
num_clusters_best = int(bestperf_params.iloc[0]['num_clusters'])

print('Loading autoencoder of Zspacedim_best = {}, mu = {} , post training DEC results'.format(Zspacedim_best,num_clusters_best))
dec_model = DECModel(mx.cpu(), X, num_clusters_best, 1.0, Zspacedim_best, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 

with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(Zspacedim_best,num_clusters_best,labeltype)), 'rb') as fu:
    dec_model = pickle.load(fu)
  
with gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_{}.arg'.format(Zspacedim_best,num_clusters_best,labeltype)), 'rb') as fu:
    outdict = pickle.load(fu)

#####################
# extract Z-space from optimal DEC model
#####################
# saved output results
dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
'encoder_3_bias', 'encoder_2_bias']
dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
dec_args['dec_mubestacci'] = dec_model['dec_mu']

N = X.shape[0]
all_iter = mx.io.NDArrayIter({'data': X}, batch_size=X.shape[0], shuffle=False,
                                          last_batch_handle='pad')   
## extract embedded point zi 
mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
aDEC = DECModel(mx.cpu(), X, num_clusters_best, 1.0, Zspacedim_best, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 

# gather best-Zspace or dec_model['zbestacci']
zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X.shape[0], aDEC.xpu).values()[0]      
zbestacci = dec_model['zbestacci']

# compute model-based best-pbestacci or dec_model['pbestacci']
pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])
pbestacci = dec_model['pbestacci']

# pool Z-space variables
datalabels = np.asarray(y)
dataZspace = np.concatenate((zbestacci, pbestacci), axis=1) 

#####################
# unbiased assessment: SPlit train/held-out test
#####################
# to compare performance need to discard unkown labels, only use known labels (only B or M)
Z = dataZspace[datalabels!='K',:]
y = datalabels[datalabels!='K']
  
print '\n... MLP fully coneected layer trained on Z_train tested on Z_test' 
sep = int(X.shape[0]*0.10)
Z_test = Z[:sep]
yZ_test = np.asanyarray(y[:sep]=='M').astype(int) 
Z_train = Z[sep:]
yZ_train = np.asanyarray(y[sep:]=='M').astype(int) 
   
# We’ll load MLP using MXNet’s symbolic interface
dataMLP = mx.sym.Variable('data')
# MLP: two fully connected layers with 128 and 32 neurons each. 
fc1  = mx.sym.FullyConnected(data=dataMLP, num_hidden = 128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 32)
act2 = mx.sym.Activation(data=fc2, act_type="relu")
# data has 2 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
# Softmax output layer
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
# create a trainable module on CPU     
batch_size = 50
mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
# pass train/test data to allocate model (bind state)
MLP_train_iter = mx.io.NDArrayIter(Z_train, yZ_train, batch_size, shuffle=False)
mlp_model.bind(MLP_train_iter.provide_data, MLP_train_iter.provide_label)
mlp_model.init_params()   
mlp_model_params = mlp_model.get_params()[0]

# update parameters based on optimal found during cv Training
from mxnet import ndarray
params_dict = ndarray.load(os.path.join(save_to,'mlp_model_params_z{}_mu{}.arg'.format(Zspacedim_best,num_clusters_best)))
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

# order of params: [(128L, 266L),(128L,),(32L, 128L),(32L,),(2L, 32L),(2L,)]
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
# upddate model parameters
mlp_model.set_params(mlp_model_params, aux_params)

#####################
# ROC: Z-space MLP fully coneected layer for classification
#####################
figROCs = plt.figure(figsize=(9,9))    
axaroc = figROCs.add_subplot(1,1,1)
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)
# Evaluate a score by cross-validation
tprs = []; aucs = []
mean_fpr = np.linspace(0, 1, 100)
cvi = 0
for train, test in cv.split(Z_train, yZ_train):
    # Pass cv data
    MLP_val_iter = mx.io.NDArrayIter(Z_train[test], None, batch_size)    
    # prob[i][j] is the probability that the i-th validation contains the j-th output class.
    prob_val = mlp_model.predict(MLP_val_iter)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(yZ_train[test], prob_val.asnumpy()[:,1])
    # to create an ROC with 100 pts
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # plot
    #axaroc.plot(fpr, tpr, lw=1, alpha=0.6) # with label add: label='cv %d, AUC %0.2f' % (cvi, roc_auc)
    cvi += 1
    
axaroc.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
axaroc.plot(mean_fpr, mean_tpr, color='b',label=r'DEC cv training (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=3, alpha=1)     

################
# plot AUC on validation set
################
MLP_heldout_iter = mx.io.NDArrayIter(Z_test, None, batch_size)   
probas_heldout = mlp_model.predict(MLP_heldout_iter)
   
# Compute ROC curve and area the curve
fpr_val, tpr_val, thresholds_val = roc_curve(yZ_test, probas_heldout.asnumpy()[:, 1])
auc_val = auc(fpr_val, tpr_val)
axaroc.plot(fpr_val, tpr_val, color='r',label=r'DEC testing (AUC = %0.2f)' % (auc_val),lw=3, alpha=1)     

### add original for comparison
axaroc.plot(mean_fpr_OrigX, mean_tpr_OrigX, color='b',
         label=r'HD training (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_OrigX, std_auc_OrigX),lw=3, ls='-.', alpha=.8)     
axaroc.plot(fpr_val_OrigX, tpr_val_OrigX, color='r',
            label=r'HD testing (AUC = %0.2f)' % (auc_val_OrigX),lw=3, ls='-.',  alpha=.8)   

axaroc.set_xlim([-0.05, 1.05])
axaroc.set_ylim([-0.05, 1.05])
axaroc.set_xlabel('False Positive Rate',fontsize=18)
axaroc.set_ylabel('True Positive Rate',fontsize=18)
axaroc.set_title('ROC LD DEC optimized space={}, all features={} - Unsupervised DEC + cv MLP classifier'.format(Z.shape[0],Z.shape[1]),fontsize=18)
axaroc.legend(loc="lower right",fontsize=18)
plt.show()

'''
##################################################################
## 5) t-SNE
################################################################## 
'''        
figtsne = plt.figure(figsize=(12,16))
axtsne = figtsne.add_subplot(1,1,1)
            
tsne = TSNE(n_components=2, perplexity=15, learning_rate=125,
     init='pca', random_state=0, verbose=2, method='exact')

Z_tsne = tsne.fit_transform(Z)   
   
plot_embedding_unsuper_NMEdist_intenh(Z_tsne, dec_model['named_y'], axtsne, title='final tsne: Acc={}\n'.format(max(dec_model['bestacci'])), legend=True)

