import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def ROCcurve(truescore, predscore):
    fpr, tpr, thresholds = roc_curve(truescore, predscore)
    roc_auc = roc_auc_score(truescore, predscore)
    ## Sensitivity and specificity
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    subscore = predscore - optimal_threshold
    subscore[subscore >= 0] = 1
    subscore[subscore < 0] = 0
    confusionMatrix = confusion_matrix(truescore, subscore)
    return fpr, tpr, thresholds, roc_auc, confusionMatrix


def DiagnosisPerformance(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    total = tn + fp + fn + tp
    Accuracy = (tp + tn) / total  # Well-predicted sample # among total sample
    Precision = tp / (tp + fp)  # Real positive among Positively predicted
    Sensitivity = tp / (tp + fn)  # Positively predicted among Real positive
    Specificity = tn / (tn + fp)  # Negatively predicted among Real negative
    return Accuracy, Precision, Sensitivity, Specificity


def genStats(trueScore, predScore):
    fpr, tpr, thresholds, roc_auc, confusionMatrix = ROCcurve(trueScore, predScore)
    Accuracy, Precision, Sensitivity, Specificity = DiagnosisPerformance(confusionMatrix)
    tnr = 1 - fpr
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    if len(tpr[np.argwhere(tnr >= 0.95)]) != 0:
        sens_at_95spec = np.max(tpr[np.argwhere(tnr >= 0.95)])
    else:
        sens_at_95spec = np.nan

    if len(tpr[np.argwhere(tnr >= 0.99)]) != 0:
        sens_at_99spec = np.max(tpr[np.argwhere(tnr >= 0.99)])
    else:
        sens_at_99spec = np.nan

    if len(tnr[np.argwhere(tpr >= 0.95)]) != 0:
        spec_at_95sens = np.max(tnr[np.argwhere(tpr >= 0.95)])
    else:
        spec_at_95sens = np.nan

    if len(tnr[np.argwhere(tpr >= 0.99)]) != 0:
        spec_at_99sens = np.max(tnr[np.argwhere(tpr >= 0.99)])
    else:
        spec_at_99sens = np.nan

    stats = pd.DataFrame(data=[roc_auc, Sensitivity, Specificity, Accuracy, Precision],
                         index=["roc_auc", "Sensitivity", "Specificity", "Accuracy", "Precision"])
    stats_senspe9599 = pd.DataFrame(data=[sens_at_95spec, sens_at_99spec, spec_at_95sens, spec_at_99sens],
                                    index=["sens_at_95spec", "sens_at_99spec", "spec_at_95sens", "spec_at_99sens"])
    fprtpr = pd.DataFrame(np.concatenate((fpr.reshape(-1, 1), tpr.reshape(-1, 1)), axis=1), columns=['FPR', 'TPR'])
    return stats, stats_senspe9599, thresholds, fprtpr


def SummaryStats_CI(result_merged, n_bootstrap = 1000, rng_seed = 777):
    ''' ROC curve '''
    trueScore = np.array((result_merged['Group'] == 'TRGT').astype(int))
    predScore = np.array(result_merged['MeanScore'])

    # Original population
    stats_op, stats_senspe9599_op, thresholds_op, fprtpr_op = genStats(trueScore, predScore)
    # Bootstrapping
    rng = np.random.RandomState(rng_seed)

    bs_stats = pd.DataFrame()
    bs_indices = pd.DataFrame()
    for i in tqdm(range(n_bootstrap)):
        indices = rng.randint(0, len(trueScore), len(predScore))
        if len(np.unique(trueScore[indices])) < 2:
            continue

        stats, stats_senspe9599, thresholds, fprtpr = genStats(trueScore[indices], predScore[indices])
        stats.columns = ['Set_' + str(i)]
        bs_stats = pd.concat([bs_stats, stats], axis=1)
        bs_indices = pd.concat([bs_indices, pd.DataFrame(indices.reshape(-1, 1), columns=['Set_' + str(i)])], axis=1)

    roclist = bs_stats.loc['roc_auc']
    roclist_95 = roclist[(roclist >= np.percentile(roclist, 2.5)) & (roclist <= np.percentile(roclist, 97.5))]

    indices_min = bs_indices[roclist_95.idxmin()]
    indices_max = bs_indices[roclist_95.idxmax()]

    stats_min, stats_senspe9599_min, thresholds, fprtpr_min = genStats(trueScore[indices_min], predScore[indices_min])
    stats_max, stats_senspe9599_max, thresholds, fprtpr_max = genStats(trueScore[indices_max], predScore[indices_max])

    merged_stats = pd.concat([stats_op, stats_min, stats_max], axis=1)
    merged_stats.columns = ['Original population', 'Min_AUC', 'Max_AUC']

    merged_stats_senspe9599 = pd.concat([stats_senspe9599_op, stats_senspe9599_min, stats_senspe9599_max], axis=1)
    merged_stats_senspe9599.columns = ['Original population', 'Min_AUC', 'Max_AUC']

    return merged_stats, merged_stats_senspe9599, fprtpr_op, fprtpr_min, fprtpr_max


