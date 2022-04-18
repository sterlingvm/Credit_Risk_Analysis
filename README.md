# Credit_Risk_Analysis

## Analysis Overview
For this project, we will be building, evaluating, and leveraging multiple supervised machine learning models to predict credit risk and overall approval of bank loans based on a variety of factors (features). We will be building these machine learning models using Python.

### Models we'll be utilizing:
Data imbalance correction algorithms:
    - Oversampling: **RandomOverSampler** algorithm & **SMOTE** algorithm.
    - Undersampling: **ClusterCentroids** algorithm.
    - Combination Sampling: **SMOTEENN** algorithm.
Classifier Models we'll be using:
    - **BalancedRandomForestClassifier** model
    - **EasyEnsembleClassifier** model

We will be evaluate the performance of these models through accuracy socres, confusion matrices, and classification reports, then we will recommend whether or not each individual model should be used to predict credit risk.

## Resources
- Data Source: LoanStats_2019Q1.csv
- Software: Python 3.8.8, Visual Studio Code 1.66.2

## Results (Balanced Accuracy Scores, Confusion Matrixes and Imbalanced Classification Reports)

### RandomOverSampler model
![RandomOverSampler](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/ros_acc.png?raw=true "RandomOverSampler")
![RandomOverSampler](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/ros_conf.png?raw=true "RandomOverSampler")
![RandomOverSampler](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/ros_class_rep.png?raw=true "RandomOverSampler")
</p>
The balanced accuracy score is 63%.<br>The high_risk precision is 1% and has a sensitivity of 60%, the F1 score is 2%.<br>The low_risk precision is 100% and has a sensitivity of 66%, the F1 score is 80%.<br>

Verdict: This model should not be used to predict credit risk and/or loan application approval or denial.
<br><br>

### SMOTE model
![SMOTE](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/smote_acc.png?raw=true "SMOTE")
![SMOTE](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/smote_conf.png?raw=true "SMOTE")
![SMOTE](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/smote_class_rep.png?raw=true "SMOTE")
</p>
The balanced accuracy score is 66%.<br>The high_risk precision is 1% and has a sensitivity of 66%, the F1 score is 2%.<br>The low_risk precision is 100% and has a sensitivity of 67%, the F1 score is 80%.<br>

Verdict: This model should not be used to predict credit risk and/or loan application approval or denial.
<br><br>

### ClusterCentroids model
![Cluster Centroids](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/cc_acc.png?raw=true "Cluster Centroids")
![Cluster Centroids](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/cc_conf.png?raw=true "Cluster Centroids")
![Cluster Centroids](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/cc_class_rep.png?raw=true "Cluster Centroids")
</p>
The balanced accuracy score is 51%.<br>The high_risk precision is 1% and has a sensitivity of 53%, the F1 score is 1%.<br>The low_risk precision is 100% and has a sensitivity of 49%, the F1 score is 66%.<br> 

Verdict: This model should not be used to predict credit risk and/or loan application approval or denial.
<br><br>

### SMOTEENN model
![SMOTEENN](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/smoteenn_acc.png?raw=true "SMOTEENN")
![SMOTEENN](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/smoteenn_conf.png?raw=true "SMOTEENN")
![SMOTEENN](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/smoteenn_class_rep.png?raw=true "SMOTEENN")
</p>
The balanced accuracy score is 65%.<br>The high_risk precision is 1% and has a sensitivity of 70%, the F1 score is 2%.<br>The low_risk precision is 100% and has a sensitivity of 61%, the F1 score is 75%.<br> 

Verdict: This model should not be used to predict credit risk and/or loan application approval or denial.
<br><br> 

### BalancedRandomForestClassifier model
![BalancedRandomForestClassifier](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/brfc_acc.png?raw=true "BalancedRandomForestClassifier")
![BalancedRandomForestClassifier](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/brfc_conf.png?raw=true "BalancedRandomForestClassifier")
![BalancedRandomForestClassifier](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/brfc_class_rep.png?raw=true "BalancedRandomForestClassifier")
</p>
The balanced accuracy score is 83%.<br>The high_risk precision is 3% and has a sensitivity of 79%, the F1 score is 6%.<br>The low_risk precision is 100% and has a sensitivity of 87%, the F1 score is 93%.<br> 

Verdict: This model can be used to predict credit risk and/or loan application approval or denial!
<br><br>

### EasyEnsembleClassifier model
![EasyEnsembleClassifier](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/eec_acc.png?raw=true "EasyEnsembleClassifier")
![EasyEnsembleClassifier](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/eec_conf.png?raw=true "EasyEnsembleClassifier")
![EasyEnsembleClassifier](https://github.com/sterlingvm/Credit_Risk_Analysis/blob/main/Resources/eec_class_rep.png?raw=true "EasyEnsembleClassifier")
</p>
The balanced accuracy score is 93%.<br>The high_risk precision is 7% and has a sensitivity of 92%, the F1 score is 12%.<br>The low_risk precision is 100% and has a sensitivity of 93%, the F1 score is 97%.<br> 

Verdict: This model can definitely be used to predict credit risk and/or loan application approval or denial!
<br><br>

## Summary
All of the models show low precision in detecting high_risk applications, but that may be - in part - due to the fact that the original data that the models were trained on have so few high risk loan applications that the model can't tell much about what would make a loan application high risk beyond the specific parameters/circumstances of the few instances od high risk in the original dataset. As such, the models have a range of high_risk precision scores between 1%-7%, but everything beyond 2% or 3% can be considered to be significantly positive increases in performance because of the lack of high risk datapoints in the original data as previously discussed.

When we run a Logistic Regression model with our 4 variations of resampling models, all of the models seem to underperform, with accuracy scores between 51%-66%. This is dangerously close to the model being as good as taking a random 50-50 guess as to whether you (the bank) should approave a loan application. This could lead to financial risk for the bank as the model wouldn't have an accurate shot of guessing whether a loan applicant would pay back the money lent or not based on previous data.
As a result a Logistic Regression Machine Learning Model implementing:
    - RandomOverSampling
    - SMOTE synthetic oversampling
    - Cluster Centroid Undersampling
    - SMOTEENN combination sampling
would not be recommended for predicting credit risk based on the data it has been trained on.

When we run our Ensemble Learning Machine Learning models, we see significant improvement in their ability to predict credit risk accurately based on the data it has been trained on - as we see accuracy scores between 83%-93%. This means that the BalancedRandomForestClassifier and - especially - the EasyEnsembleCLassifier would be fantastic predictors for predicting the credit risk of loan applications based on the data it was trained on.
As a result:
    - BalancedRandomForestClassifier
    - EasyEnsembleClassifier
would be well recommended for predicting credit risk based on the data it has been trained on.