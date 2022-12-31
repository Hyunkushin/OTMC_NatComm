# OTMC_NatComm

## Overview
 These codes were written for peer review of a scientific paper <One Test-Multi Cancer: simultaneous, early detection of multi-cancer using Liquid Biopsy based on Exosome-SERS-AI> and demo of the results. Unauthorized use of this code for other purpose is prohibited.
 This repository contains Python codes to show example of cancer detector and TOO detector using exosomal SERS signals. In `./DataBase_demo` directory, sample data for demo of the codes are included. Also, codes for drawing main figures and calculate diagnostic performance are included.
 
## System requirement
 - Python 3.8.8
 - Tensorflow 2.5.0
 - Pandas 1.4.2
 - Scikit-learn 0.24.1
 - Matlab R2021a
  All python codes are recommended using python IDE (e.g. PyCharm, Spyder)

## Source data and models
 This repository includes source data for reproduction of the main figures.
 The source data containing the numerial values for the figures is stored as an excel file in `./data_generator/source_data` directory.
 The implemented and optimized models for cancer diagnosis, TOO discrimination, and multi-layer perceptron is stored in the same directory.

## Reproduction of data
<img src="https://img.shields.io/badge/Python-FFCA28?style=flat-square&logo=Python&logoColor=000000"/> <img src="https://img.shields.io/badge/MatLab R2021a-ED7D31?style=flat-square&logo=MatLab R2021a&logoColor=111111"/>

 All `.py` or `.m` files starting with 'fig' in the `./data_generator` directory are codes written for re-implementation of figure.
 If you run the code through the recommended Python IDE or MatLab, you can check the original figure data.
 
 `extract_ROC_CI.py` is a file to calculate statistics for quantifying the diagnostic performance of our model.
 This code generates the output, including AUC of ROC, sensitivity, specificity, accuracy, and precision value with a 95% confidence interval.
 This code offers the major result on cancer presence detection (Table 1) and TOO discrimination (Table 2).
 CI can be derived differently for each trial depending on the random seed value (Default = 777), and the repeated sampling number for bootstrapping was set to 1000.

## Decision maker
<img src="https://img.shields.io/badge/Python-FFCA28?style=flat-square&logo=Python&logoColor=000000"/>

`decision_maker.py` is a programmed code that shows an example of operation to identify both cancer presence and TOO detection.
This decision algorithm is based on the pretrained models and weights in `./data_generator/source_data` directory.
The output will be displayed in console with ID, true label, and prediction result, like below. 'LUAD', 'BRCA', 'COAD', 'LIHC', 'PAAD', and 'STAD' are assigned to Lung, Breast, Colon, Liver, Pancreas, and Stomach cancer, respectively.

- Output example
<pre><code>[ID: 1Col13]
True label: COAD
==> Prediction: Cancer detected.
    TOO decision: COAD (Correct)</code></pre>


