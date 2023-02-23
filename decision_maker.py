"""
This code is for reproducing results of a scientific paper 'Hyunku Shin, et al., <Single test-based early diagnosis of
multiple cancer types using Exosome-SERS-AI>'. Unauthorized use for other purpose is prohibited.
"""

import numpy as np
import pandas as pd
import pickle
import glob2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# TF
import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


class Classifier():
    motherPath = r".\data_generator\source_data"
    def __init__(self, modelFolderName):
        self.model = tf.keras.models.load_model(Classifier.motherPath + '/' + modelFolderName)
    def predict(self, spectra):
        return self.model.predict(spectra)
    def predict_mean(self, spectra):
        return np.mean(self.model.predict(spectra))


def Load_Classifiers():
    """ Load classifier """
    CD_clf = Classifier('model_CD')
    TOO_clf = dict()
    for lbl in ['LUAD', 'BRCA', 'COAD', 'LIHC', 'PAAD', 'STAD']:
        TOO_clf[lbl] = Classifier('model_TOO/' + lbl + '_detector')
    MLP_clf = Classifier('model_MLP')
    with open(r".\data_generator\source_data\model_MLP" + "/enc.pkl", 'rb') as f: MLP_enc = pickle.load(f)
    return CD_clf, TOO_clf, MLP_clf, MLP_enc

def Decision_maker(spectra):
    """ Predict sample """
    cutoff_CD_data = 0.50
    cutoff_CD = 0.85
    pred_idvd = CD_clf.predict(spectra)
    if np.mean(pred_idvd) > cutoff_CD:
        spectra_passed = spectra.iloc[list(np.where(pred_idvd > cutoff_CD_data)[0]), -800:]
        score_TOO = np.zeros((1, 6))
        for j, lbl in enumerate(['LUAD', 'BRCA', 'COAD', 'LIHC', 'PAAD', 'STAD']):
            score_TOO[0, j] = TOO_clf[lbl].predict_mean(spectra_passed)
        score_TOO = pd.DataFrame(score_TOO, columns=['LUAD', 'BRCA', 'COAD', 'LIHC', 'PAAD', 'STAD'])
        decision = MLP_enc.inverse_transform(MLP_clf.predict(score_TOO))[0][0]
    else:
        score_TOO = pd.DataFrame(np.zeros((1,6)), columns=['LUAD', 'BRCA', 'COAD', 'LIHC', 'PAAD', 'STAD'])
        decision = 'CNTL'
    return decision, np.mean(pred_idvd), score_TOO

if __name__ == "__main__":

    """ Load Model """
    CD_clf, TOO_clf, MLP_clf, MLP_enc = Load_Classifiers()

    """ DB prediction (demo) """
    DBdir = r".\DataBase_demo"
    DBfiles = glob2.glob(DBdir + '/*.pickle')
    for i, db_dir in enumerate(DBfiles):
        with open(db_dir, 'rb') as f:
            db = pickle.load(f)
        spectra = db.iloc[:, -800:]
        info = db[['ID', 'true_label']]
        decision, score_CD, score_TOO = Decision_maker(spectra)

        print('\n[ID: %s]'  %info['ID'][0])
        print('True label: %s' %info['true_label'][0])
        if decision == info['true_label'][0]:
            check = 'Correct'
        else:
            check = 'Incorrect'
        if score_CD > 0.85:
            print('==> Prediction: Cancer detected.')
            print('    TOO decision: %s (%s)' %(decision, check))
        else:
            print('==> Prediction: Cancer not detected.')


