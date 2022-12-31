"""
This code is for reproducing results of a scientific paper 'Hyunku Shin, et al., <One Test-Multi Cancer: simultaneous, early detection of
multi-cancer using Liquid Biopsy based on Exosome-SERS-AI>'. Unauthorized use for other purpose is prohibited.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import precision_recall_curve, auc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def fill_and_reshape(map, fix, axis=0):
    if len(map)%fix != 0:
        filler = np.zeros(fix - len(map)%fix)
        filler[:] = np.nan
    else:
        filler = np.array([])
    if axis == 0:
        reshapedMap = np.concatenate((map, filler)).reshape(-1, fix)
    elif axis == 1:
        reshapedMap = np.concatenate((map, filler)).reshape(-1, fix).T
    else:
        print('[Caution] "axis" should be 0 (col) or 1 (row). The value is set to 0 now.')
        reshapedMap = np.concatenate((map, filler)).reshape(-1, fix)
    bin = np.zeros((40, 40))
    bin[:] = np.nan
    bin[:reshapedMap.shape[0], :reshapedMap.shape[1]] = reshapedMap
    return bin


plt.close('all')

# ''' Heatmap '''
# for smpl in ['train', 'test']:
#     # Data
#     sourceData = pd.read_excel(r".\data_generator\source_data"+ '/Source Data.xlsx', sheet_name = 'F3_Score value_' + smpl)
#     labelabbre = ['CNTL', 'LUAD', 'BRCA', 'COAD', 'LIHC', 'PAAD', 'STAD']
#     preset = [5, 5, 5, 5, 5, 5, 5]
#     # Figure
#     for i, lbl in enumerate(labelabbre):
#         fig = plt.figure(figsize=(12, 10))
#         plt.rc('font', size=8, family='arial')
#         fig.tight_layout()
#         predictedMap = sourceData[sourceData['Label_abbreviation'] == lbl]['MeanScore']
#         sns.heatmap(pd.DataFrame(fill_and_reshape(predictedMap, preset[i], 1)), cmap='viridis', vmin=0, vmax=1,
#                     xticklabels=False, yticklabels=False, linewidth=2)
#         plt.title(lbl)
        # plt.savefig(r".\data_generator\fig_data" + '/Fig3_c_Heatmap_' + smpl + '_' + lbl + '.png', dpi=150)

if __name__ == "__main__":
    ''' Violine plot '''
    # Data
    sourceData = pd.read_excel(r".\source_data"+ '/Source Data.xlsx', sheet_name = 'F3_Score value_test')
    # Figure
    fig = plt.figure(figsize=(6, 5))
    fig.tight_layout()
    plt.rc('font', size=12.5, family='arial')
    labelfile = ['Lun', 'Bre', 'Col', 'Liv', 'Pan', 'Sto']
    labelabbre = ['CNTL', 'LUAD', 'BRCA', 'COAD', 'LIHC', 'PAAD', 'STAD']
    colors = ['#878787', '#C57A59', '#4D977C', '#BE374C', '#9572A7', '#C8AA6E', '#49848C']
    ax = sns.violinplot(data=sourceData, x="Label_abbreviation", y="MeanScore", order=labelabbre, palette=colors,
                        scale='width', inner='box', linewidth=0.5)
    plt.ylabel('Mean score')
    plt.xlabel('')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['HC', 'Lung', 'Breast', 'Colon', 'Liver', 'Pancreas', 'Stomach'])
    plt.xticks(rotation=0)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.show()
    # if you want to save the figures
    # plt.savefig(r".\data_generator\fig_data" + "/Fig3_d_Violin plot.png", dpi=300)
    # plt.savefig(r".\data_generator\fig_data" + "/Fig3_d_Violin plot.svg", format = 'svg')

    ''' ROC curve '''
    # Data
    sourceData = pd.read_excel(r".\source_data"+ '/Source Data.xlsx', sheet_name = 'F3_ROC curve')
    # Figure
    fig_roc = plt.figure(figsize=(5.2, 5))
    plt.rc('font', size=12.5, family='arial')
    fig_roc.tight_layout()
    plt.fill(sourceData['FPR_CI_MAX'], sourceData['TPR_CI_MAX'], color='r', alpha=0.3)
    plt.fill(sourceData['FPR_CI_MIN'], sourceData['TPR_CI_MIN'], color='w')
    plt.plot(sourceData['FPR_P'], sourceData['TPR_P'], 'r-')
    plt.plot([0, 1], [0, 1], 'k:')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.show()
    # if you want to save the figures
    # plt.savefig(r".\data_generator\fig_data" + "/Fig3_e_ROC curve.png", dpi=300)
    # plt.savefig(r".\data_generator\fig_data" + "/Fig3_e_ROC curve.eps", format = 'eps')

    ''' ROC curve by cancer type '''
    # Data
    labels = ['Lun', 'Bre', 'Col', 'Liv', 'Pan', 'Sto']
    colors = ['#C57A59', '#4D977C', '#BE374C', '#9572A7', '#C8AA6E', '#49848C']
    sourceData = pd.read_excel(r".\source_data"+ '/Source Data.xlsx', sheet_name = 'F3_ROC curve by type')
    # Figure
    fig_bt = plt.figure(figsize=(5.2, 5))
    fig_bt.tight_layout()
    plt.rc('font', size=12.5, family='arial')
    for i, lbl in enumerate(labels):
        plt.plot(sourceData['FPR_' + lbl], sourceData['TPR_' + lbl], color = colors[i])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
    plt.legend(['Lung (0.936)', 'Breast (0.984)', 'Colon (0.972)', 'Liver (0.978)', 'Pancreas (0.992)', 'Stomach (0.999)'],
               loc=4, prop={'size': 15})
    plt.plot([0, 1], [0, 1], 'k:')
    plt.show()
    # if you want to save the figures
    # plt.savefig(r".\data_generator\fig_data" + "/Fig3_f_ROC curve by type.png", dpi=300)
    # plt.savefig(r".\data_generator\fig_data" + "/Fig3_f_ROC curve by type.eps", format = 'eps')

    ''' PRC curve '''
    # Data
    sourceData = pd.read_excel(r".\source_data"+ '/Source Data.xlsx', sheet_name = 'F3_Score value_test')
    trueScore = np.array((sourceData['Group'] == 'TRGT').astype(int))
    predScore = np.array(sourceData['MeanScore'])
    precision, recall, thresholds = precision_recall_curve(trueScore, predScore)
    prc_auc = auc(recall, precision)
    f1_scores = 2 * recall * precision / (recall + precision)
    print('Best threshold: ', thresholds[np.argmax(f1_scores)])
    print('Best F1-Score: ', np.max(f1_scores))
    # Figure
    fig_prc = plt.figure(figsize=(5.2, 5))
    fig_prc.tight_layout()
    plt.rc('font', size=12.5, family='arial')
    plt.plot(recall, precision, 'b-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([min(recall), 1.02])
    plt.ylim([min(precision), 1.005])
    plt.title('F1-value = ' + str(round(np.max(f1_scores), 3)) + '  /  prc_auc = ' + str(round(prc_auc, 3)))
    plt.text(0.2, 0.2, str(np.max(f1_scores)))
    plt.legend(['Precision-recall curve (area = %0.3f)' % round(prc_auc, 3)])
    plt.show()
    # if you want to save the figures
    # plt.savefig(r".\data_generator\fig_data" + "/Fig3_g_PRC curve.png", dpi=300)
    # plt.savefig(r".\data_generator\fig_data" + "/Fig3_g_PRC curve.eps", format = 'eps')