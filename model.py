import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("diabe.csv")

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)

scaler=StandardScaler()
colum_to_scale=["Pregnancies" ,"Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
scaler.fit(df[colum_to_scale])
df[colum_to_scale]=scaler.transform(df[colum_to_scale])
X = df.drop("Outcome", axis=1)
Y = df["Outcome"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.25,random_state=42, stratify=Y)


random_forest = RandomForestClassifier()

param_test1 = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1200,num=12)],
    'criterion' : ['gini', 'entropy'],
    'max_depth':[5,10,15,20,25,30],
    'min_samples_leaf' : [1,2,5, 10],
    'min_samples_split' : [1,2, 10,25]
}
rsearch1 = RandomizedSearchCV(estimator = random_forest,param_distributions = param_test1,scoring='roc_auc', cv=5)

rsearch1.fit(X_train, Y_train)



file=open('random_forest_diabetes_prediction.pkl','wb')
pickle.dump(rsearch1,file)
file1=open('scaler.pkl','wb')
pickle.dump(scaler,file1)
file1.close()