# Home-Credit-Scoring
- This is my personal project on working with tabular data. It contains both codes and literature review. Due to limitations of resouces, I could not manage to train with all data (greater than 1.5 millions instances. Instead, I resampled 305,000 instances after multicollinearity filtering with VIF. These instanced were used for feature selection and training model for later tasks 

## Evironment and Packages
Python 3.9

To install the packages:
```
pip install requirement.txt
```
## Dataset 

The dataset from the kaggle competition [Home credit: credit risk model stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data) 

## Workflow
### Training

1. Train model
```
python main.py --setup train
```

2. Evaluate model
```
python eval.py
```
## Literature reivew written by me
[A survey on credit scoring models](https://drive.google.com/file/d/1tlTC5S4XQWLIZ59qmVK6IE7gFR3NDMfr/view?usp=sharing)
