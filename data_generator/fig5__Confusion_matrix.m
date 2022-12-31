% 22-12-15
% This code is for reproducing results of a scientific paper
% 'Hyunku Shin, et al., <One Test-Multi Cancer: simultaneous,
% early detection of multi-cancer using Liquid Biopsy based on
% Exosome-SERS-AI>'.
% Unauthorized use for other purpose is prohibited.

clc; clear all; close all;
mat = xlsread('.\data_generator\source_data\Source Data.xlsx', 'F5_Confusion matrix')

h = heatmap(mat);
h.ColorScaling = 'scaledrows';
h.XData = ["Non-cancer" "Lung" "Breast" "Colon" "Liver" "Pancreas" "Stomach"]
h.YData = flip(["Non-cancer" "Lung" "Breast" "Colon" "Liver" "Pancreas" "Stomach"])
xlabel('Predicted class')
ylabel('Actual class')
set(gcf,'units','points','position',[100,100,350,300]);