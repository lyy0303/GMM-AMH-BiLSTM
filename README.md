# 1. Toolbox:
### (1) The Data folder contains the original data of the MIT-BIH Arrhythmia Database.
### (2) DBtool.py is used to read data and locate feature points. The returned attributes include the record number the data belongs to, the original signal, the filtered signal, frequency, R-peak positions, labels, and other information.

# 2. DataProcess:
### (1) heart_beats_segment.py is used for waveform segmentation.

| wavetype | boundary |
|----------|----------|
| P wave   | ≤ 300 ms |
| Q wave   | ≤ 120 ms |
| S wave   | ≤ 180 ms |
| T wave   | ≥ 300 ms |
# 3. GaussECG:
### (1) Fit 5 waveforms using 5 Gaussian components.
<img src="/DataAnalysis/100_0_ecg.png" alt="Example of partition results" style="width: 800px; height: auto;">

# 4. DataAnalysis:
### (1) Build the AMH-BiLSTM model.
### (2) Process data and merge partitioned datasets.
### (3) Import data and train and test.
<img src="/DataAnalysis/confusion_matrix_AMHBiLSTM.png" alt="Confusion matrix" style="width: 400px; height: auto;">
<img src="/DataAnalysis/ROC_curve_AMHBiLSTM.png" alt="ROC curve" style="width: 400px; height: auto;">
