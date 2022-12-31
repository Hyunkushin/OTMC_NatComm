"""
This code is for reproducing results of a scientific paper 'Hyunku Shin, et al., <One Test-Multi Cancer: simultaneous, early detection of
multi-cancer using Liquid Biopsy based on Exosome-SERS-AI>'. Unauthorized use for other purpose is prohibited.

This code is to calculate statistical value about prediction performance and accuracy,
including sensitivity, specificity, accuracy, precision, and AUC of ROC with 95% confidence interval (CI).
The data for statistics are based on the prediction results of the test samples ('result_testMerged.csv' of each 'source_data' folder).
"""

import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from core.getStats import SummaryStats_CI


labelabbre = ['LUAD', 'BRCA', 'COAD', 'LIHC', 'PAAD', 'STAD'] # Lung, Breast, Colon, Liver, Pancreas, Stomach

''' Cancer detector '''
path = r".\source_data\model_CD"
result = pd.read_csv(path + '/result_testMerged.csv')

# HC vs all cancer
merged_stats_entire, merged_stats_senspe9599_entire, fprtpr_op_entire, fprtpr_min_entire, fprtpr_max_entire = SummaryStats_CI(result, n_bootstrap = 1000, rng_seed = 777)
print('\n----------------------------------------------------')
print('HC vs All cancer')
print(merged_stats_entire)
print('----------------------------------------------------')

fig_roc = plt.figure(figsize=(5.2, 5))
plt.rc('font', size=12, family='arial')
fig_roc.tight_layout()
plt.fill(fprtpr_max_entire['FPR'], fprtpr_max_entire['TPR'], color='r', alpha=0.3)
plt.fill(fprtpr_min_entire['FPR'], fprtpr_min_entire['TPR'], color='w')
plt.plot(fprtpr_op_entire['FPR'], fprtpr_op_entire['TPR'], 'r-')

# HC vs each cancer
merged_stats = dict()
merged_stats_senspe9599 = dict()
fprtpr_op = dict()
fprtpr_min = dict()
fprtpr_max = dict()
for lbl in labelabbre:
    result_selected = result[result['Label_abbreviation'].isin(['CNTL', lbl])]
    merged_stats[lbl], merged_stats_senspe9599[lbl], fprtpr_op[lbl], fprtpr_min[lbl], fprtpr_max[lbl] = SummaryStats_CI(result_selected, n_bootstrap = 1000, rng_seed = 777)
    print('\n----------------------------------------------------')
    print('HC vs ' +lbl)
    print(merged_stats[lbl])
    print('----------------------------------------------------')


''' TOO detector '''
merged_stats = dict()
merged_stats_senspe9599 = dict()
fprtpr_op = dict()
fprtpr_min = dict()
fprtpr_max = dict()
for lbl in labelabbre:
    path = r".\source_data\model_TOO" + '/' + lbl + '_detector'
    result = pd.read_csv(path + '/result_testMerged.csv')
    merged_stats[lbl], merged_stats_senspe9599[lbl], fprtpr_op[lbl], fprtpr_min[lbl], fprtpr_max[lbl] = SummaryStats_CI(result, n_bootstrap = 1000, rng_seed = 777)
    print('\n----------------------------------------------------')
    print(lbl + ' detector')
    print(merged_stats[lbl])
    print('----------------------------------------------------')


