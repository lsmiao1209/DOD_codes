# DOD_codes

Official repository for the paper
"Self-supervised enhanced denoising diffusion for anomaly detection", Information Sciences. [INS 2024](https://doi.org/10.1016/j.ins.2024.120612)

"Prototype-oriented hypergraph representation learning for anomaly detection in tabular data", Information Processing and Management. [IPM 2025](https://doi.org/10.1016/j.ipm.2024.103877)

This repository is not yet completed, so please check this as a reference only.

## Software Requirement
```
numpy== 1.25.0
torch==1.13.0+cu116
python==3.9.18
sklearn==0.22.0
pandas==2.1.3
cuda=11.6
pyod==1.0.9
adbench==0.1.11
deepod==0.4.1
torch-geometric==2.4.0
```

Below is the link to the graph neural network-related code utilized in our paper. The implementation of the fundamental neural network in our method relies on the following codebase. Please follow the instructions provided in the link to install the necessary dependencies:
[DeepHypergraph Code Repository](https://github.com/iMoonLab/DeepHypergraph/tree/main)

## About the datasets

Download data and put them into the folder: datasets/.

kddcup99 dataset is derived from the UCI KDD 10% dataset.[Datasets source](http://kdd.ics.uci.edu/databases/kddcup99/)

others are from the latest anomaly detection benchmark study, ADBench. [Datasets source](https://github.com/Minqi824/ADBench/tree/main/adbench/datasets)

## Evaluation measures
For the evaluation metrics, we utilized two mainstream methods widely employed in anomaly detection: the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) and the Area Under the Precision Recall Curve (AUC-PR).

