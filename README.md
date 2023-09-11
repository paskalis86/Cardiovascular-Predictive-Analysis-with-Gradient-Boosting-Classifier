# Cardiovascular predictive analysis using Gradient-Boosting Classifier

## Overview
Based on the data realesed by American Heart Association, in 2020 there are 928,741 death casues by cardiovascular and spent total costs $407.3 billion. There for, The Behavioral Risk Factor Surveillance System (BRFSS) United States, they were surveyed each individual about the behavior that potentially to cardiovascular disease. Their goal is to give preventive services for people who have potential cardiovascular disease. 
So, I created a machine learening to predict. Does someone have the potential for cardiovascular or not, so the BRFSS United States could give preventive services right on target and reduce costs of CVD. At the end of the result, the perfomance of model is not really good, there is overfitting on precission metric but good on recall metric with score 0.78. Since the precision is low than recall, it's mean that the BRFSS United States need more effort to filtering which is the really true positve CVD.
On the simulation of a model, if there are 100 people with CVD  and the total mean direct medical care costs for patients with established cardiovascular disease (CVD) was  18,953 US dollar  per patient per year. The model caught 78 people of them and we could save $1.5 million  of costs

## Background Story
Cardiovascular disease (CVD), listed as the underlying cause of death, accounted for 928,741 deaths in the United States in 2020. Between 2017 and 2020, 127.9 million US adults had some form of CVD. Between 2018 and 2019, direct and indirect costs of total CVD were $407.3 billion. The data based on information of [American Heart Association](https://professional.heart.org/en/science-news/-/media/453448D7D79948B39D5851D1FF2A0CFE.ashx)

The Behavioral Risk Factor Surveillance System (BRFSS) United States, they were surveyed each individual about the behavior that potentially to cardiovascular disease. Their goal is to give preventive services for people who have potential cardiovascular disease. 

The goal of these project is :
Build a prediction model, Does someone have the potential for cardiovascular or not, so the BRFSS United States could give preventive services right on target and reduce costs of CVD.
## Research Question 
1. How to make a model to predict cardiovascular disease
2. Which is the best model to use prediction of cardiovascular disease
3. What is the importance of the features that influence the model to make a prediction? 

## About the data set
I used a data set from [Kaggle](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset) to train and test the model. The data set contains 18 features, 1 target and 308854 row 

## Exploratory Data Analysis
Overall, there’s no null value on the data set. However there’s a duplicate value, and we delete the duplicates because it will make the model confuse. Remain 308695 row.

After cleaning the data set. Then I check the target and I found that 8 % of data set have cardiovascular.

![cdv 1](https://github.com/paskalis86/Cardiovascular-Predictive-Analysis-with-Gradient-Boosting-Classifier/assets/138757072/5d0644ac-0048-4649-8b06-7ef6da7fc5e0)


The target (Heart_Disease) is imbalance, this is not good to learn by the model because the model would be more understand negative than positive cardiovascular. So, we need some treatment for balancing the target. In This case, I would like to use under-sampling. 

here is a statistical summary for numeric column.

![cdv 2](https://github.com/paskalis86/Cardiovascular-Predictive-Analysis-with-Gradient-Boosting-Classifier/assets/138757072/e6d16545-5574-4a2f-9c00-f5080d720db6)


I found the abnormal value for height and weight body. The minimum height is 91 cm, whereas the minimum age is 18 years old. Than we look at outliers through box plot in below

![cdv 3](https://github.com/paskalis86/Cardiovascular-Predictive-Analysis-with-Gradient-Boosting-Classifier/assets/138757072/f8a558a6-809d-4527-a4c6-d9584ce507ab)


There are 124263 rows containing outliers. This amount is equivalent to 40 % of the row in the data set. I decided to delete all these outliers for modeling.
I use IQR method to separate which outlier or not
Outliers :
If value <  (Q1 – 1.5(IQR) ) or
    value > (Q3 +1.5(IQR)) 
Where :
Q1 = first quartile of data,
Q3 = third quartile of data,
And 
IQR = Q3 – Q1

After delete all outliers, the row remaining 184432

## Data Preparation 
Since machine learning just understands numeric values, I convert column objects into numeric ones.That column which type of object has 2 unique values ( yes/no ) I convert into (1 and 0 ). Where 1 = Yes, and 0 = No.For column like General_Health, Checkup, Age_Category, Weight_Category I treated ordinal encoding

I Split the data set 90 % for training and 10 % for validation, then in the training set I treated under-sampling with a ratio minority of 80 %. So the final data set for the training model is 31806 rows and for validation 18444 rows. The validation data set still represents the original condition of the target. 8 % of the data has cardiovascular.

![cdv 4](https://github.com/paskalis86/Cardiovascular-Predictive-Analysis-with-Gradient-Boosting-Classifier/assets/138757072/e5d1902b-b48d-44ee-b6ed-3d966d74da6c)


## Baseline model
To choose which is the best model, I did it on the baseline model. Where the features learned by the model is just numeric column, without feature engineering treatment. In other words, just the way it is.
The model that I tested is :
1. Logistic Regression
2. Decision-Tree Classifier
3. Random-Forest Classifier
4. Gradient-Boosting Classifier
![cdv 5](https://github.com/paskalis86/Cardiovascular-Predictive-Analysis-with-Gradient-Boosting-Classifier/assets/138757072/68e2d87a-447a-4189-a4d7-b5f6600bf5e4)


The reference metric to choose a model is recall and F1 Score, Based on these metric The best model is the gradient boosting classifier.

## Modeling and Evaluation
Since gradient boosting is the best model based on base line. Next, I find the best parameter to improve the metric (Hypertuning parameter). The parameter is n_estimators and max_depth.

![cdv 6](https://github.com/paskalis86/Cardiovascular-Predictive-Analysis-with-Gradient-Boosting-Classifier/assets/138757072/367b1dbf-237f-4426-9593-5d3d42e80372)

Best model parameter: n_esimator = 30 and max_depth = 5

Here is the result of training and testing 

![cdv 7](https://github.com/paskalis86/Cardiovascular-Predictive-Analysis-with-Gradient-Boosting-Classifier/assets/138757072/37991230-b0ac-4ff8-a3fe-1f0642868a9e)


interpretate model :
1. best model is on parameter n_estimators : 30 and max_depth : 5
2. precision metric has an overfitting, where training model higher than validation
3. recall is metric we focus on. from all occurrence heart disease, the model caught 78 %
4. precision metric 23 % is mean that, from all true positive prediction made by the  model. only 23 % that true positive
#### Interpretation of model performance based on validation data set
The picture below is a confusion matrix. It represents the result of model prediction compared with actual data.
1. True Positive (TP) : 1241 row or 6,72 %
2. True Negative (TN) : 12637 row or 68,51 %
3. False Positive (FP) : 4208 row or 22,81 %
4. False Negative (FN) : 358 row or 1,94 %

![cdv 8](https://github.com/paskalis86/Cardiovascular-Predictive-Analysis-with-Gradient-Boosting-Classifier/assets/138757072/061b808f-9647-4376-9d57-c2e8cf162008)


The below picture is the representation distribution from the result of the model prediction.
- Circle I : All row of validation data
- Yellow   : True Negative
- Circle II and red   : The row that predicted as False Negative. We lost them because they are positive cardiovascular.
- Circle III          : All row who predicted as True Positive
- Green               : False Positive (FP)
- Circle IV and blue : The row which is the true positive (TP)

![cdv 9](https://github.com/paskalis86/Cardiovascular-Predictive-Analysis-with-Gradient-Boosting-Classifier/assets/138757072/4e34e613-8fb5-4c8f-a687-6620422d9d4c)


## Estimation potential impact of model
For example, if there are 100 out of 300 people who are sick with CVD, 78 people from them will be predicted as true positive. But, because since the precision metric is 23 %, there are a lot of people who will predicted as positive by the model, which will confuse

According to a study conducted in USA, the total mean direct medical care costs for patients with established cardiovascular disease (CVD) was  18,953 US dollar  per patient per year with inpatient costs being 42.8%  ($8114) of total costs 

so, if we can prevent of 78 person who has potential cardiovascular. we could save $1.5 million  of costs

## Feature Importance
![cdv 10](https://github.com/paskalis86/Cardiovascular-Predictive-Analysis-with-Gradient-Boosting-Classifier/assets/138757072/afd2b0b3-635d-417e-b58d-f8bf28dd6078)


observation : 
1. features importance in above is global feature importance
2. Top feature mostly is categorical data, such as age category, general health, check-up, sex, diabetes, smoking history, and arthritis.
3. Feature importance on the model does not represent of correlation to the target, but we can associate those as features that most contribute for model to predict true or false.
4. consumption behavior is low contribution for model to predict 

## Conclusion 
1. The model for prediction of cardiovascular disease has been made with a Gradient-boosting classifier. Since there are a lot of outlier in the numeric column, I decided to delete all outliers because it is abnormal dat
2. The gradient Boosting classifier has been chosen as the best model based on the baseline model, with parameter n_estimators = 30, max_depth = 5. With performance recall = 0.78, precision = 0.23, and F1 score = 0.35, since the target has imbalance classes it causes overfitting on precision metric
3. The top feature is important to the target is age category, general health, check-ups, sex, diabetes, smoking history, and arthritis. Which is all of these factors is similar to factor of cardiovascular disease

## Recommendation
Since precision metric is low, here is some of recommendation action :

1. use this model as the first filtering for predictive, If someone predicted by the model is positive cardiovascular then take a medical chekup by medical for make sure, it is true positif or not. The advantage of using this model, you can save time in checking all patients on the data set, you just need to check 29 % of patients from the data set.
2. consultation with a specialist in cardiovascular about the features, we still can explore new features that contain the behavior of the patient, make it more detailed and relevan to target. So, the model could be more accurate
3. About feature importance, don't use this as reference of factors that causes cardiovascular, instead investigation and observation of each patient is needed for further
4. Investigate more further of outliers, especially weight and height body. It is valid or not
5. Give education about cardiovascular to people, any factor that impact to cardiovascular and how to reduce the potential





