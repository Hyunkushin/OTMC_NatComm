% 22-12-15
% This code is for reproducing results of a scientific paper
% 'Hyunku Shin, et al., Single test-based early diagnosis of
% multiple cancer types using Exosome-SERS-AI'.
% Unauthorized use for other purpose is prohibited.

clc; clear all; close all;
mat = xlsread('.\source_data\Source Data.xlsx', 'F5_Confusion matrix')

h = heatmap(mat);
h.ColorScaling = 'scaledrows';
h.XData = ["Non-cancer" "Lung" "Breast" "Colon" "Liver" "Pancreas" "Stomach"]
h.YData = flip(["Non-cancer" "Lung" "Breast" "Colon" "Liver" "Pancreas" "Stomach"])
xlabel('Predicted class')
ylabel('Actual class')
set(gcf,'units','points','position',[100,100,350,300]);