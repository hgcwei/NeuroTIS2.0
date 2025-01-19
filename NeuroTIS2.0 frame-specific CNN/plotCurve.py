import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import auc,roc_curve
from sklearn.metrics import f1_score,matthews_corrcoef,precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import matplotlib.colors as mcolors

def eval_perf(label, scores):

    fpr, tpr, thresholds = roc_curve(label, scores)
    pr, re, thresholds2 = precision_recall_curve(label, scores)
    for i in range(len(thresholds)):
        if thresholds[i] < 0.5:
            sn, sp, auROC, auPRC = tpr[i], 1 - fpr[i], auc(fpr, tpr), auc(re, pr)
            break

    scoress = []
    for i in range(len(scores)):
        if scores[i]>0.5:
            scoress.append(1)
        else:
            scoress.append(0)

    label = np.int8(np.array(label))
    label = label.tolist()
    pre = precision_score(label,scoress)
    acc = accuracy_score(label,scoress)
    f1 = f1_score(label, scoress)
    mcc = matthews_corrcoef(label, scoress)
    perf_scores = (sn, sp, pre, acc, f1, auROC, auPRC, mcc)
    curve_scores = (fpr, tpr, re, pr)

    return perf_scores, curve_scores

class PlotCurve:

    def __init__(self,score_ls, label_ls, legends_ls):

        self.score_ls = score_ls
        self.label_ls = label_ls
        self.legends_ls = legends_ls
        self.curve_scores_ls = []

        self.colors = [mcolors.CSS4_COLORS[color] for color in mcolors.CSS4_COLORS]
        self.numbers = [x for x in range(len(self.colors))]
        random.shuffle(self.numbers)

    def set_curve_scores_ls(self):
        for i in range(len(self.score_ls)):
            c, psls = eval_perf(self.label_ls[i], self.score_ls[i])
            print(c)
            self.curve_scores_ls.append(psls)

    def plot_roc(self):
        fig, ax = plt.subplots()
        for i in range(len(self.curve_scores_ls)):
            (fpr, tpr, re, pr) = self.curve_scores_ls[i]
            ax.plot(fpr, tpr, color=self.colors[self.numbers[i]])

        # add axis labels to plot
        ax.set_title('True and False Positive Curve')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        plt.legend(self.legends_ls)
        plt.xlim([0, 0.15])
        plt.ylim([0.85, 1])
        # display plot
        plt.show()