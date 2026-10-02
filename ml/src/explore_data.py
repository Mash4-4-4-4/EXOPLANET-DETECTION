import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
df=pd.read_csv("data/kepler_data.csv")#reads teh csv file

features = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_impact",
    "koi_incl",
    "koi_prad",
    "koi_sma",
    "koi_ror",
    "koi_dor",
    "koi_teq",
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass"
]

corr = df[features].corr()

plt.figure(figsize=(12, 10))
sb.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
    
label_encoder=LabelEncoder() #creating an instance of the label encoder
x=df[features]
print("Features describe:", x.describe()) #describes the features in the dataset
y=df["koi_disposition"]

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20,random_state=42,stratify=y)
imputer=SimpleImputer(strategy="median") #we are using the median strategy to fill in missing values    

X_train=imputer.fit_transform(X_train) #we are fitting the imputer to the training data and transforming it only on training data and not on testing data to avoid data leakage. We are using the median strategy to fill in missing values.
X_test=imputer.transform(X_test) #we are transforming the testing data using the fitted imputer to avoid data leakage. We are using the median strategy to fill in missing values.
Y_train=label_encoder.fit_transform(Y_train) #we are fitting the label encoder to the training data and transforming it only on training data and not on testing data to avoid data leakage. We are using the label encoder to convert categorical labels into numerical labels.
Y_test=label_encoder.transform(Y_test)#
print(label_encoder.classes_)

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=Y_train
)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

xgb=XGBClassifier(objective="multi:softprob",
                  eval_metric="mlogloss",
                  random_state=42)

random_search=RandomizedSearchCV(
    estimator=xgb,#which model are we tuning
    param_distributions=param_grid,#what parameters are we tuning on
    n_iter=20,#try randomly selected 20 combinations of paramteres
    scoring="f1_macro",#use f1 macro score when comparing different models and rounds
    cv=5,#means 5 fold cross validation forex-(for f1 trains on others f2,f3....and test only on f1 continues for every f)
    random_state=42,
    n_jobs=-1#makes it faster teh cpu will use all the cores available to it
)

random_search.fit(X_train,Y_train,
                  sample_weight=sample_weights) #fitting the random search to the training data
model=random_search.best_estimator_ #getting the best model from the random search
print(random_search.best_params_) #printing the best parameters found by the random search
print(random_search.best_score_) #printing the best score found by the random search

print("*********************************************************")
# model=XGBClassifier(n_estimators=200,
#                     max_depth=5,#controls how deep each tree can go 
#                     learning_rate=0.05, #controls how much to change the model in response to the estimated error each time the model weights are updated
#                     subsample=0.8, #controls the fraction of samples that are randomly chosen to grow each tree
#                     colsample_bytree=0.8, #controls the fraction of features that are
#                     random_state=42) #creating xgboost model with n_estimators(no of trees)

# model.fit(X_train,Y_train,sample_weight=sample_weights) #fitting the model to the training dat
feature_importance=pd.DataFrame({
    "feature":features,
    "importance":model.feature_importances_
})
impo=feature_importance.sort_values(by="importance",
                                    ascending=False)


Y_pred=model.predict(X_test)#making predictions on the testing data
Y_prob=model.predict_proba(X_test) #predicting the probabilities of each class for the testing data
roc_auc=roc_auc_score(Y_test,Y_prob,multi_class="ovr") #calculating the roc auc score for the testing data

sb.heatmap(confusion_matrix(Y_test,Y_pred),annot=True,fmt="d")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("confusion matrix")


print(impo)
print("accuracy:",accuracy_score(Y_test,Y_pred))
print("confusion_matrix:",confusion_matrix(Y_test,Y_pred))
print("classification:",classification_report(Y_test,Y_pred)) #this gives precision recall f1 score support (it calculates the models performance for each class)
print("ROC AUC Score:", roc_auc)

print(df["koi_disposition"].value_counts())
print(df["koi_disposition"].value_counts(normalize=True))

