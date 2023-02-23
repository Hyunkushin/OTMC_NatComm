"""
This code is for reproducing results of a scientific paper 'Hyunku Shin, et al., Single test-based early diagnosis of
multiple cancer types using Exosome-SERS-AI'. Unauthorized use for other purpose is prohibited.
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from core.getStats import SummaryStats_CI

if __name__ == "__main__":
    ''' Box plot '''
    # Data
    sourceData = pd.read_excel(r".\source_data" + '/Source Data.xlsx', sheet_name='F4_Box plot')
    # Figure
    labelfile = ['Lun', 'Bre', 'Col', 'Liv', 'Pan', 'Sto']
    labelabbre = ['LUAD', 'BRCA', 'COAD', 'LIHC', 'PAAD', 'STAD']
    colors = ['#C57A59', '#4D977C', '#BE374C', '#9572A7', '#C8AA6E', '#49848C']
    fig = plt.figure(figsize=(5, 15))
    fig.tight_layout()
    plt.rc('font', size=12, family='arial')
    for i, lbl in enumerate(labelabbre):
        plt.subplot(6,1,i+1)
        data = sourceData[['ID_'+lbl, 'Label_'+lbl, 'MeanScore_'+lbl]]
        sns.boxplot(data=data, x="Label_"+lbl, y="MeanScore_"+lbl, order=labelabbre, palette=colors,
                    fliersize=2, showcaps=True, width=0.4, linewidth=0.5,
                    flierprops={"marker": ".", "alpha": 0.5})
        plt.ylabel('Score')
        if not i == 5:
            plt.xlabel('')
            plt.tick_params(bottom = False, labelbottom = False)

        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')
    plt.show()
    # if you want to save the figures
    # plt.savefig(r".\data_generator\fig_data" + "/Fig4_a_Box plot.png", dpi=300)
    # plt.savefig(r".\data_generator\fig_data" + "/Fig4_a_Box plot.svg", format = 'svg', dpi=300)

    ''' ROC curve '''
    # Data
    sourceData = pd.read_excel(r".\source_data" + '/Source Data.xlsx', sheet_name='F4_ROC curve by type')
    # Figure
    fig_roc = plt.figure(figsize=(5.2, 5))
    plt.rc('font', size=12, family='arial')
    fig_roc.tight_layout()
    for i, lbl in enumerate(labelabbre):
        plt.plot(sourceData['FPR_'+lbl], sourceData['TPR_'+lbl], color = colors[i])
    plt.plot([0, 1], [0, 1], 'k:')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.show()
    # if you want to save the figures
    # plt.savefig(r".\data_generator\fig_data" + "/Fig4_b_ROC curve by type.png", dpi=300)
    # plt.savefig(r".\data_generator\fig_data" + "/Fig4_b_ROC curve by type.eps", format='eps')


    ''' ROC curve (Early-stage) '''
    # Data
    labelabbre = ['LUAD', 'BRCA', 'COAD', 'LIHC', 'PAAD', 'STAD']
    sourceData = pd.read_excel(r".\source_data" + '/Source Data.xlsx', sheet_name='F4_Box plot')
    info = pd.read_excel(r".\source_data" + '/Sample info.xlsx')
    info = info.set_index('ID')

    result_entire = dict()
    for lbl in labelabbre:
        section = sourceData[['ID_'+lbl, 'Label_'+lbl, 'MeanScore_'+lbl]]
        section.columns = ['ID', 'Label', 'MeanScore']
        section = section.set_index('ID')
        result_entire[lbl] = pd.concat([section, info], axis=1)

    # early-stage
    merged_stats = dict()
    merged_stats_senspe9599 = dict()
    fprtpr_op = dict()
    fprtpr_min = dict()
    fprtpr_max = dict()
    for lbl in labelabbre:
        result_too = result_entire[lbl]
        result_TRGT = result_too[(result_too['Stage'].isin([0, 1, 2, 'BCLC_0', 'BCLC_1'])) &
                                 (result_too['Label'] == lbl)]
        result_CNTL = result_too[result_too['Label'] != lbl]
        result = pd.concat([result_TRGT, result_CNTL])

        result.loc[result[result['Label'] == lbl].index, 'Group'] = 'TRGT'
        result.loc[result[result['Label'] != lbl].index, 'Group'] = 'CNTL'
        merged_stats[lbl], merged_stats_senspe9599[lbl], fprtpr_op[lbl], fprtpr_min[lbl], fprtpr_max[lbl] = SummaryStats_CI(result, n_bootstrap = 1000, rng_seed = 777)

    # Figure
    fig_roc = plt.figure(figsize=(7, 11))
    plt.rc('font', size=12, family='arial')
    colors = ['#C57A59', '#4D977C', '#BE374C', '#9572A7', '#C8AA6E', '#49848C']
    for i, lbl in enumerate(labelabbre):
        plt.subplot(3,2,i+1)
        plt.fill(fprtpr_max[lbl]['FPR'], fprtpr_max[lbl]['TPR'], color=colors[i], alpha=0.3)
        plt.fill(fprtpr_min[lbl]['FPR'], fprtpr_min[lbl]['TPR'], color='w')
        plt.plot(fprtpr_op[lbl]['FPR'], fprtpr_op[lbl]['TPR'], color = colors[i])
        plt.plot([0, 1], [0, 1], 'k:')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xticks([0, 0.5, 1])
        plt.yticks([0, 0.5, 1])
    plt.show()

    # plt.savefig(r".\data_generator\fig_data" + "/Fig4_c_ROC curve_early stage.png", dpi=300)
    # plt.savefig(r".\data_generator\fig_data" + "/Fig4_c_ROC curve_early stage.svg", format='svg', dpi=300)