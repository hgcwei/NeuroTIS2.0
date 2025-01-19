import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import auc,roc_curve
from sklearn.metrics import f1_score,matthews_corrcoef,precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import string
import random

def random_string(length:int) -> str:
    """
    length: 指定随机字符串长度
    """
    random_str = ''
    # base_str = string.digits
    # base_str = string.ascii_letters
    base_str = string.digits + string.ascii_letters
    for i in range(length):
        random_str += random.choice(base_str)
    return random_str
def load_scores_labels(file_sc,file_la):
    scores = np.loadtxt(file_sc, delimiter=',', dtype=np.float32)
    labels = np.loadtxt(file_la, delimiter=',', dtype=np.int)

    inds1 = (np.where(labels == 0))
    inds0 = (np.where(labels > 0))
    labels[inds1] = 1
    labels[inds0] = 0


    # score = scores[:, 0]


    return scores,labels



scores0,labels0 = load_scores_labels('neurotis+results\human\scores0.csv','neurotis+results\human\label0.csv')
scores1,labels1 = load_scores_labels('neurotis+results\human\scores1.csv','neurotis+results\human\label1.csv')
scores2,labels2 = load_scores_labels('neurotis+results\human\scores2.csv','neurotis+results\human\label2.csv')

print(scores0.shape)
np.savetxt('scores_neurotis_plus_g.csv',np.vstack((scores0,scores1,scores2)),delimiter=',')



scores_ = scores0.tolist() + scores1.tolist() + scores2.tolist()
labels_ = labels0.tolist() + labels1.tolist() + labels2.tolist()
np.savetxt('labels_neurotis_plus_g.csv',[ 1-item for item in labels_],delimiter=',')
# np.savetxt('scores_neurotis_plus_g.csv',scores_,delimiter=',')
scores, labels = load_scores_labels('neurotis ng results\human\scores_neurotis_plus_ng.csv', 'neurotis ng results\human\labels_neurotis_plus_ng.csv')

scoresT,labelsT = load_scores_labels('TISRover\human\scoresTISRover.csv','TISRover\human\labelTISRover.csv')
scoresT2,labelsT2 = load_scores_labels('NeuroTIS\human\scoresNeuroTISNone.csv','NeuroTIS\human\labelNeuroTISNone.csv')


scoresT3 = np.loadtxt('TITER\human\scores1.csv', delimiter=',', dtype=np.float32)
labelsT3 = np.loadtxt('TITER\human\labels1.csv', delimiter=',', dtype=np.int)
labelsT3 = labelsT3.tolist()


import plotCurve

prp = plotCurve.PlotCurve([scores_,scores,scoresT2,scoresT,scoresT3],[labels_,labels,labelsT2,labelsT,labelsT3],['NeuroTIS+ (G)','NeuroTIS+ (nG)','NeuroTIS','TISRover','TITER'])
prp.set_curve_scores_ls()
prp.plot_roc()
