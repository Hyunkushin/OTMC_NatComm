"""
This code is for reproducing results of a scientific paper 'Hyunku Shin, et al., Single test-based early diagnosis of
multiple cancer types using Exosome-SERS-AI'. Unauthorized use for other purpose is prohibited.
"""

import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    sourceData = pd.read_excel(r".\source_data" + '/Source Data.xlsx', sheet_name='F2_SERS signal')
    xaxis = np.array(sourceData.iloc[1:,0]).astype("float32")
    data_mean = np.array(sourceData.iloc[1:,1:8]).astype("float32")
    data_std = np.array(sourceData.iloc[1:, 10:17]).astype("float32")

    plt.close('all')
    fig = plt.figure(figsize=(4, 7))
    plt.rc('font', size=12, family='arial')
    colors = ['#323335', '#C57A59', '#4D977C', '#BE374C', '#9572A7', '#C8AA6E', '#49848C']
    for i in range(7):
        SERS_mean = data_mean[:,i]
        SERS_std = data_std[:,i]

        plt.plot(xaxis, SERS_mean + (6-i), color=colors[i])
        plt.fill_between(xaxis, SERS_mean - SERS_std + (6-i), SERS_mean + SERS_std + (6-i),
                         alpha=0.6, edgecolor='w', facecolor=colors[i], linewidth=0)
        plt.xlim([540, 1750])
        plt.xticks(np.linspace(600, 1600, 6))
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    fig.tight_layout()
    plt.show()

    # fig.savefig(r".\data_generator\fig_data" + "/Fig2_f_Raman signals.png", format='png', dpi = 300)  # Save PNG Imag
    # fig.savefig(r".\data_generator\fig_data" + "/Fig2_f_Raman signals.eps", format='eps')  # Save Vector Image