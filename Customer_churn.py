#### Preprocess Data and build classifier for telecom based Customer Churn Data ####
#### Program written by Kushal Mahalingaiah 
#### contact @kushalhosahallimahalingaiah@gmail.com
### Import required packages for preprocessing and analysis
"""
Package versions:
python        : 3.6.5
scitkit-learn : 0.19.1
pandas        : 0.23.0
numpy         : 1.14.3
matplotlib    : 2.2.2
seaborn       : 0.8.1
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes 
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     RandomizedSearchCV)
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)

### Import data into pandas as dataframe
df = pd.read_csv('churn_dataset.csv')
#displaying head of loaded dataset
print("---Dataset Loaded---")
print(df.head(5))
#drop rows that have all NA values
df.dropna()

### Get preliminary info about dataframe
print("---Dataset Structure info---")
print(df.info()) # Categorical variables are of type 'object'
# print(df.shape)
print("---Null values---")
print(df.isnull().sum()) # counting number of NaNs

## Define list of categorical variables
cat_vars = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'Churn']

## Print value counts of categorical variables and replace NA with most occuring one
print("---Value counts for each feature---")
for var in cat_vars:
    print(df[var].value_counts())
    df[var] = df[var].fillna(df[var].value_counts().argmax())

## Set TotalCharges to float 
df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

#replace NA with mean value for sum/value based column
df.tenure = df.tenure.fillna(df.tenure.mean())
df.MonthlyCharges = df.MonthlyCharges.fillna(df.MonthlyCharges.mean())
df.TotalCharges = df.TotalCharges.fillna(df.TotalCharges.mean())

#checking NA values after filling missing/NA values
# print(df.isnull().sum()) # No NaNs

## Set senior citizen to type category based on 0 or 1
df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'yes', 0: 'no'}).astype('category')

## Check info of continuous features 
# print(df.info()) # Conversion successful
# print(df.describe()) # Disparate ranges, should be normalized

### Examine continuous features
## Examine correlations
df['tenuremonth'] = (df['tenure'] * df['MonthlyCharges']).astype(float)
df.corr()


## Collapse categories where appropriate
#  Multiple lines 'no phone service' to 'no'
df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 'No',
                                               'Yes': 'Yes',
                                               'No': 'No'}).astype('category')
df['MultipleLines'].value_counts() 

#  6 features, convert 'no internet service' to 'no'
no_int_service_vars = ['OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection','TechSupport', 
                       'StreamingTV', 'StreamingMovies']

for var in no_int_service_vars:
    df[var] = df[var].map({'No internet service': 'No',
                           'Yes': 'Yes',
                           'No': 'No'}).astype('category')
    
# for var in no_int_service_vars:
#     print(df[var].value_counts())
    
#uncomment plt.show() lines if you want to see plots while executing
## Plot distributions of categorical variables
for var in cat_vars:
    ax = sns.countplot(x = df[var], data = df, palette = 'colorblind')
    total = float(len(df[var])) 
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 10,
                '{:1.2f}'.format(height/total),
                ha="center")
    plt.title('Distribution of ' + str(var))
    plt.ylabel('Number of Customers')
    plt.figtext(0.55, 0.01, 
                'Decimal above bar is proportion of total in that class',
                horizontalalignment = 'center', fontsize = 8,
                style = 'italic')
    plt.xticks(rotation = 60)
    plt.tight_layout()
    plt.savefig('plot_dist-' + str(var) + '.png', dpi = 200)
    # plt.show()
    plt.clf()
    plt.cla()
    plt.close()

### Exploratory Plots
## Churn by tenure
plt.subplot(1,2,1)
sns.violinplot(x = df['Churn'], y = df['tenure'], data = df, inner = None,
               palette = 'colorblind')
plt.title('Churn by Customer Tenure')
# plt.tight_layout()
# plt.savefig('plot-churn_by_customers.png', dpi = 200)
# plt.show()

## Churn by monthly charges
plt.subplot(1,2,2)
sns.violinplot(x = df['Churn'], y = df['MonthlyCharges'], data = df, inner = None,
               palette = 'colorblind')
plt.title('Churn by Monthly Charge')
plt.tight_layout()
plt.savefig('plot-churn_by_charges_tenure.png', dpi = 200)
# plt.show()
plt.clf()
plt.cla()
plt.close()

## Churn by contract length
contract_churn = df.groupby(['Contract', 'Churn']).agg({'customerID': 'count'})
contract = df.groupby(['Contract']).agg({'customerID': 'count'})
contract_churn_pct = contract_churn.div(contract, level='Contract') * 100
contract_churn_pct = contract_churn_pct.reset_index()

sns.barplot(x = 'Contract' , y = 'customerID', hue = 'Churn',
            data = contract_churn_pct)
plt.title('Churn Rate by Contract Length')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_contract.png', dpi = 200)
# plt.show()
plt.clf()
plt.cla()
plt.close()

## Churn by paperless billing
PaperlessBilling_churn = df.groupby(['PaperlessBilling', 'Churn']).agg({'customerID': 'count'})
PaperlessBilling = df.groupby(['PaperlessBilling']).agg({'customerID': 'count'})
PaperlessBilling_churn_pct = PaperlessBilling_churn.div(PaperlessBilling, level='PaperlessBilling') * 100
PaperlessBilling_churn_pct = PaperlessBilling_churn_pct.reset_index()

sns.barplot(x = 'PaperlessBilling' , y = 'customerID', hue = 'Churn',
            data = PaperlessBilling_churn_pct)
plt.title('Churn Rate by PaperlessBilling type')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_PaperlessBilling.png', dpi = 200)
# plt.show()
plt.clf()
plt.cla()
plt.close()

## Churn by PaymentMethod
PaymentMethod_churn = df.groupby(['PaymentMethod', 'Churn']).agg({'customerID': 'count'})
PaymentMethod = df.groupby(['PaymentMethod']).agg({'customerID': 'count'})
PaymentMethod_churn_pct = PaymentMethod_churn.div(PaymentMethod, level='PaymentMethod') * 100
PaymentMethod_churn_pct = PaymentMethod_churn_pct.reset_index()

sns.barplot(x = 'PaymentMethod' , y = 'customerID', hue = 'Churn',
            data = PaymentMethod_churn_pct)
plt.title('Churn Rate by PaymentMethod')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_PaymentMethod.png', dpi = 200)
# plt.show()
plt.clf()
plt.cla()
plt.close()

## Churn by TechSupport
TechSupport_churn = df.groupby(['TechSupport', 'Churn']).agg({'customerID': 'count'})
TechSupport = df.groupby(['TechSupport']).agg({'customerID': 'count'})
TechSupport_churn_pct = TechSupport_churn.div(TechSupport, level='TechSupport') * 100
TechSupport_churn_pct = TechSupport_churn_pct.reset_index()

sns.barplot(x = 'TechSupport' , y = 'customerID', hue = 'Churn',
            data = TechSupport_churn_pct)
plt.title('Churn Rate by TechSupport')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_TechSupport.png', dpi = 200)
# plt.show()
plt.clf()
plt.cla()
plt.close()

## Churn by gender
gender_churn = df.groupby(['gender', 'Churn']).agg({'customerID': 'count'})
gender = df.groupby(['gender']).agg({'customerID': 'count'})
gender_churn_pct = gender_churn.div(gender, level='gender') * 100
gender_churn_pct = gender_churn_pct.reset_index()

sns.barplot(x = 'gender' , y = 'customerID', hue = 'Churn',
            data = gender_churn_pct)
plt.title('Churn Rate by Gender')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_gender.png', dpi = 200)
# plt.show()
plt.clf()
plt.cla()
plt.close()

## Churn by internet service
internet_churn = df.groupby(['InternetService', 'Churn']).agg({'customerID': 'count'})
internet = df.groupby(['InternetService']).agg({'customerID': 'count'})
internet_churn_pct = internet_churn.div(internet, level='InternetService') * 100
internet_churn_pct = internet_churn_pct.reset_index()

sns.barplot(x = 'InternetService' , y = 'customerID', hue = 'Churn',
            data = internet_churn_pct)
plt.title('Churn Rate by Internet Service Status')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_internet.png', dpi = 200)
# plt.show()
plt.clf()
plt.cla()
plt.close()

## Churn by phone service
phone_churn = df.groupby(['PhoneService', 'Churn']).agg({'customerID': 'count'})
phone = df.groupby(['PhoneService']).agg({'customerID': 'count'})
phone_churn_pct = phone_churn.div(phone, level='PhoneService') * 100
phone_churn_pct = phone_churn_pct.reset_index()


sns.barplot(x = 'PhoneService' , y = 'customerID', hue = 'Churn',
            data = phone_churn_pct)
plt.title('Churn Rate by Phone Service Status')
plt.ylabel('Percent of Class')
plt.tight_layout()
plt.savefig('plot-churn_by_phone.png', dpi = 200)
# plt.show()
plt.clf()
plt.cla()
plt.close()


### Standardize continuous variables for distance based models
scale_vars = ['tenure', 'MonthlyCharges']
scaler = StandardScaler() 
df[scale_vars] = scaler.fit_transform(df[scale_vars])
df[scale_vars].describe()


### Drop ID and TotalCharges vars
df = df.drop(['customerID', 'TotalCharges', 'tenuremonth'],  axis = 1)
# print(df.info())

### Encode data for analyses
## Binarize binary variables
df_enc = df.copy()
binary_vars = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'PhoneService', 'MultipleLines', 'OnlineSecurity', 
               'OnlineBackup','DeviceProtection', 'TechSupport', 
               'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
               'Churn']
enc = LabelEncoder()
df_enc[binary_vars] = df_enc[binary_vars].apply(enc.fit_transform)

## One-hot encode multi-category cat. variables
multicat_vars = ['InternetService', 'Contract', 'PaymentMethod']
df_enc = pd.get_dummies(df_enc, columns = multicat_vars)
df_enc.iloc[:,16:26] = df_enc.iloc[:,16:26].astype(int)
# print(df_enc.info())


## Change categorical variables to type 'category', reduces memory size by >1/2
df[cat_vars] = df[cat_vars].astype('category')
# print(df.info()) # Conversion successful

print(time.localtime)

### Split into training data
X = df_enc.drop('Churn', axis = 1)
y= df_enc['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3,
                                                    stratify = y,
                                                    random_state = 30)

print("----Executing models----")
### KNN Model
## Start time
print("----KNN----")
knn_start = time.time()

## Instantiate classifier

knn = KNeighborsClassifier()

## Set up hyperparameter grid for tuning
knn_param_grid = {'n_neighbors' : np.arange(5,26),
                  'weights' : ['uniform', 'distance']}

## Tune hyperparameters
knn_cv = GridSearchCV(knn, param_grid = knn_param_grid, cv = 5)

## Fit knn to training data
knn_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned KNN Parameters: {}".format(knn_cv.best_params_))
print("Best KNN Training Score: {}".format(knn_cv.best_score_)) 

## Predict knn on test data
print("KNN Test Performance: {}".format(knn_cv.score(X_test, y_test)))

## Obtain model performance metrics
knn_pred_prob = knn_cv.predict_proba(X_test)[:,1]
knn_auroc = roc_auc_score(y_test, knn_pred_prob)
print("KNN AUROC: {}".format(knn_auroc))
knn_y_pred = knn_cv.predict(X_test)
print(classification_report(y_test, knn_y_pred))

## End time
knn_end = time.time()
print("KNN execution time in seconds - ",knn_end - knn_start)

### Fit logistic regression model
## Start time
print("----LogisticRegression----")
lr_start = time.time()

## Instantiate classifier
lr = LogisticRegression(random_state = 30)

## Set up hyperparameter grid for tuning
lr_param_grid = {'C' : [0.0001, 0.001, 0.01, 0.05, 0.1] }

## Tune hyperparamters
lr_cv = GridSearchCV(lr, param_grid = lr_param_grid, cv = 5)

## Fit lr to training data
lr_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned LR Parameters: {}".format(lr_cv.best_params_))
print("Best LR Training Score:{}".format(lr_cv.best_score_)) 

## Predict lr on test data
print("LR Test Performance: {}".format(lr_cv.score(X_test, y_test)))

## Obtain model performance metrics
lr_pred_prob = lr_cv.predict_proba(X_test)[:,1]
lr_auroc = roc_auc_score(y_test, lr_pred_prob)
print("LR AUROC: {}".format(lr_auroc))
lr_y_pred = lr_cv.predict(X_test)
print(classification_report(y_test, lr_y_pred))

## End time
lr_end = time.time()
print("Logistic regression execution time in seconds - ",lr_end - lr_start)

### Random Forest (RF)
## Start time
print("----RandomForestClassifier----")
rf_start = time.time()

## Instatiate classifier
rf = RandomForestClassifier(random_state = 30)

## Set up hyperparameter grid for tuning
rf_param_grid = {'n_estimators': [200, 250, 300, 350, 400, 450, 500],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4]}

## Tune hyperparameters
rf_cv = RandomizedSearchCV(rf, param_distributions = rf_param_grid, cv = 5, 
                           random_state = 30, n_iter = 20)

## Fit RF to training data
rf_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned RF Parameters: {}".format(rf_cv.best_params_))
print("Best RF Training Score:{}".format(rf_cv.best_score_)) 

## Predict RF on test data
print("RF Test Performance: {}".format(rf_cv.score(X_test, y_test)))

## Obtain model performance metrics
rf_pred_prob = rf_cv.predict_proba(X_test)[:,1]
rf_auroc = roc_auc_score(y_test, rf_pred_prob)
print("RF AUROC: {}".format(rf_auroc))
rf_y_pred = rf_cv.predict(X_test)
print(classification_report(y_test, rf_y_pred))

## Inspect feature importances
rf_optimal = rf_cv.best_estimator_
rf_feat_importances = pd.Series(rf_optimal.feature_importances_, 
                             index=X_train.columns)
rf_feat_importances.nlargest(5).plot(kind='barh', color = 'r')
plt.title('Feature Importances from Random Forest Classifier')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.savefig('model-rf_feature_importances.png', dpi = 200,
            	bbox_inches = 'tight')
# plt.show()
plt.clf()
plt.cla()
plt.close()

## End time
rf_end = time.time()
print("Random Forest execution time in seconds - ",rf_end - rf_start)

### GradientBoostingClassifier (SGB) model
## Start time
print("----Stochastic GradientBoostingClassifier----")
sgb_start = time.time()

## Instantiate classifier
sgb = GradientBoostingClassifier(random_state = 30)

## Set up hyperparameter grid for tuning
sgb_param_grid = {'n_estimators' : [200, 300, 400, 500],
                  'learning_rate' : [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
                  'max_depth' : [3, 4, 5, 6, 7],
                  'min_samples_split': [2, 5, 10, 20],
                  'min_weight_fraction_leaf': [0.001, 0.01, 0.05],
                  'subsample' : [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  'max_features': ['sqrt', 'log2']}

## Tune hyperparamters
sgb_cv = RandomizedSearchCV(sgb, param_distributions = sgb_param_grid, cv = 5, 
                            random_state = 30, n_iter = 20)

## Fit SGB to training data
sgb_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned SGB Parameters: {}".format(sgb_cv.best_params_))
print("Best SGB Training Score:{}".format(sgb_cv.best_score_)) 

## Predict SGB on test data
print("SGB Test Performance: {}".format(sgb_cv.score(X_test, y_test)))

## Obtain model performance metrics
sgb_pred_prob = sgb_cv.predict_proba(X_test)[:,1]
sgb_auroc = roc_auc_score(y_test, sgb_pred_prob)
print("SGB AUROC: {}".format(sgb_auroc))
sgb_y_pred = sgb_cv.predict(X_test)
print(classification_report(y_test, sgb_y_pred))

## Inspect feature importances
sgb_optimal = sgb_cv.best_estimator_
sgb_feat_importances = pd.Series(sgb_optimal.feature_importances_, 
                                 index=X_train.columns)
sgb_feat_importances.nlargest(5).plot(kind='barh', color = 'r')
plt.title('Feature Importances from Stochastic Gradient Boosting Classifier')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.savefig('model-sgb_feature_importances.png', dpi = 200,
            	bbox_inches = 'tight')
# plt.show()
plt.clf()
plt.cla()
plt.close()

## End time
sgb_end = time.time()
print("Stochastic Gradient Boosting execution time in seconds - ",sgb_end - sgb_start)

### Plot ROC for all models
knn_fpr, knn_tpr, knn_thresh = roc_curve(y_test, knn_pred_prob)
plt.plot(knn_fpr,knn_tpr,label="KNN: auc="+str(round(knn_auroc, 3)),
         color = 'blue')

lr_fpr, lr_tpr, lr_thresh = roc_curve(y_test, lr_pred_prob)
plt.plot(lr_fpr,lr_tpr,label="LR: auc="+str(round(lr_auroc, 3)),
         color = 'red')

rf_fpr, rf_tpr, rf_thresh = roc_curve(y_test, rf_pred_prob)
plt.plot(rf_fpr,rf_tpr,label="RF: auc="+str(round(rf_auroc, 3)),
         color = 'green')

sgb_fpr, sgb_tpr, sgb_thresh = roc_curve(y_test, sgb_pred_prob)
plt.plot(sgb_fpr,sgb_tpr,label="SGB: auc="+str(round(sgb_auroc, 3)),
         color = 'yellow')

plt.plot([0, 1], [0, 1], color='gray', lw = 1, linestyle='--', 
         label = 'Random Guess')
# plt.subplot(1,2,1)
plt.legend(loc = 'best', frameon = True, facecolor = 'lightgray')
plt.title('ROC Curve for Classification Models')
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.text(0.85,0.75, 'threshold = 0', fontsize = 8)
plt.arrow(0.85,0.8, 0.14,0.18, head_width = 0.01)
plt.text(0.05,0, 'threshold = 1', fontsize = 8)
plt.arrow(0.05,0, -0.03,0, head_width = 0.01)
plt.tight_layout()
plt.savefig('plot-ROC_4models.png', dpi = 300)
# plt.show()
plt.clf()
plt.cla()
plt.close()