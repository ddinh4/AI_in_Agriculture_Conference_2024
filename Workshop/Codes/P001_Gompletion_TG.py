# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:46:16 2023

@author: agentimis1
"""
#%%
# Basic packages
import pandas as pd 
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Preprocessing packages
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import  LabelEncoder
from sklearn.impute import SimpleImputer

# Model packages
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Accuracy Packages
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score



#=== Unused packages
#from sklearn.metrics import classification_report
#from keras import models, layers
#from tensorflow.keras.layers import Dropout



#%% Sets the working directory to the main directory containing the project
dir1=os.path.realpath(__file__)
main = os.path.dirname(os.path.dirname(dir1))
#%% Reads the dataframe
for k in range(10,15):
 
 SEM=str(k)
 print("Working on Semester "+SEM)    
 Ex0= pd.read_csv(main+'/Data/ModelReady/Completion/Graduation_Semester_'+SEM+'.csv',low_memory=False)
#%% Imputes, using most frequent and creates the Ex2 dataframe
 imputer = SimpleImputer(strategy='most_frequent')
 Ex1=pd.DataFrame(imputer.fit_transform(Ex0), columns=Ex0.columns)
 cols = list(Ex1)
 cols.insert(0, cols.pop(cols.index('COMPLETED')))
 Ex2=Ex1.loc[:,cols]
#%% Splits to input and output
 X=Ex2.iloc[:,1:len(Ex2.columns)].values
 y=Ex2.iloc[:,0].values.flatten()

#%% Extracts the number of levels of the output variable
 num_levels = Ex2.iloc[:,0].nunique()
#%% Sets up a parameter estimation grid
 param_grid = {
    'n_estimators': [10, 200],
    'max_depth': [ 30, 50],
    'max_features': [ 'sqrt', 'log2']}

#%% Performs Grid search using all data
 grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
 grid_search.fit(X, y)
#%% Computes best parameters and the corresponding classifier
 best_params = grid_search.best_params_

#%% Initialize variables to store best model information
 best_rf_model = None
 best_rf_accuracy = 0.0

 best_lr_model = None
 best_lr_accuracy = 0.0

 best_nb_model = None
 best_nb_accuracy = 0.0

 best_xgb_model = None
 best_xgb_accuracy = 0.0
#%% Inititalize vectors to store the accuracies and set the number of repetitions
 rf_accuracies= []
 lr_accuracies=[]
 nb_accuracies = []
 xgb_accuracies = []

 reps=5
#%% Scales and encodes the data
 label_encoder = LabelEncoder()
 y_encoded = label_encoder.fit_transform(y)
# Standardize the features
 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(X)

#%% Main Loop
 for i in range(reps):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=None)
    
    # Random forest classifier with best parameters
    rf = RandomForestClassifier(**best_params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred)
    rf_accuracies.append(rf_accuracy)
    if rf_accuracy > best_rf_accuracy:
        best_rf_accuracy = rf_accuracy
        best_rf_model = rf
    
    # Multi class logistic regression
    lr = LogisticRegression(max_iter=1000, multi_class='multinomial') 
    lr.fit(X_train, y_train)
    lr_y_pred = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_y_pred)
    lr_accuracies.append(lr_accuracy)
    if lr_accuracy > best_lr_accuracy:
        best_lr_accuracy = lr_accuracy
        best_lr_model = lr
    
    #Naive Bayes Classifier
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_y_pred = nb.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_y_pred)
    nb_accuracies.append(nb_accuracy)
    if nb_accuracy > best_nb_accuracy:
        best_nb_accuracy = nb_accuracy
        best_nb_model = nb
        
    # XG Boost
    xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=num_levels)
    xgb_classifier.fit(X_train, y_train)
    xgb_y_pred = xgb_classifier.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
    xgb_accuracies.append(xgb_accuracy)
    if xgb_accuracy > best_xgb_accuracy:
        best_xgb_accuracy=xgb_accuracy
        best_xgb_model = xgb_classifier     
        
    print("Finished Itteration",i)
#%% Saving best model RF and XGBoost
 joblib.dump(best_rf_model, main+'\Results\F02_Graduation\Semester'+SEM+'\Models\RandomForest.pkl')   
#joblib.dump(best_xgb_model, main+'\Results\Models\Categories\XGBoost.pkl')
#%% Compiles all the accuracies in a dataframe and saves it
 Results=pd.DataFrame({'Random_Forest':rf_accuracies,"Regression":lr_accuracies,'Bayesian':nb_accuracies,'XGBoost':xgb_accuracies})
 Results.to_csv(main+'\Results\F02_Graduation\Semester'+SEM+'\Accuracies\Accuracy.csv')
#%% Computes and Reports mean Accuracies
 mean_rf_accuracy = sum(rf_accuracies) / len(rf_accuracies)
 mean_lr_accuracy = sum(lr_accuracies) / len(lr_accuracies)
 mean_nb_accuracy = sum(nb_accuracies) / len(nb_accuracies)
 mean_xgb_accuracy = sum(xgb_accuracies) / len(xgb_accuracies)

 print("Mean Random Forest Accuracy:", mean_rf_accuracy)
 print("Mean Multinomial Logistic Regression Accuracy:", mean_lr_accuracy)
 print("Mean Gaussian Naive Bayes Accuracy:", mean_nb_accuracy)
 print("Mean XGBoost Accuracy:", mean_xgb_accuracy)
#%% Plotting Accuracies
 fig, ax = plt.subplots()
 ax.boxplot(Results)
 ax.set_title('Side by Side Boxplot of Accuracies for different Models')
 ax.set_xlabel('Predictive Models')
 ax.set_ylabel('Accuracies')
 xticklabels=['Random Forest','Multilevel Regression','Naive Bayes','XGBoost']
 ax.set_xticklabels(xticklabels)
 ax.yaxis.grid(True)
 plt.savefig(main+'/Results/F02_Graduation/Semester'+SEM+'/Graphs/Side_by_Side_Graduation.png')

#%% Showing and saving the confusion matrix for the best RF
 y_pred = best_rf_model.predict(X_test)
 conf_matrix = confusion_matrix(y_test, y_pred)
 levels = np.unique(y_test)
 conf_matrix_df = pd.DataFrame(conf_matrix, index=levels, columns=levels)
 print("Confusion Matrix:")
 print(conf_matrix_df)
 conf_matrix_df.to_csv(main+'/Results/F02_Graduation/Semester'+SEM+'/Confusion/Conf_Matrix_Graduation_RF.csv')
#%% Computes the most important features in the RF
 feature_importances=best_rf_model.feature_importances_
 most_important_indices = feature_importances.argsort()[:][::-1]  # Change 10 to the desired number of features
 Important_Input=Ex1.iloc[:,most_important_indices]
 most_important_features=Important_Input.columns.tolist()
#%% Plot feature importances for the RF
 plt.figure(figsize=(10, 6))
 plt.barh(range(len(most_important_features)), feature_importances[most_important_indices], align='center')
 plt.yticks(range(len(most_important_features)), most_important_features)
 plt.xlabel('Feature Importance')
 plt.ylabel('Features')
 plt.title('Most Significant Predictors')
 plt.savefig(main+"/Results/F02_Graduation/Semester"+SEM+"/Graphs/FeatureImportance_Categories_Year0.png")

#%% Perform t-SNE for dimensionality reduction with 2 components
 tsne = TSNE(n_components=2, random_state=42)
 reduced_data = tsne.fit_transform(X_scaled)

#%% Combine reduced data with target variable
 reduced_df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2'])
 reduced_df['COMPLETED'] = y

#%% Plot t-SNE clusters
 plt.figure(figsize=(10, 8))
 sns.scatterplot(x='Component 1', y='Component 2', hue='COMPLETED', data=reduced_df, palette='tab10', legend='full')
 plt.title('t-SNE Visualization of Graduation')
 plt.xlabel('Component 1')
 plt.ylabel('Component 2')
 plt.savefig(main+"/Results/F02_Graduation/Semester"+SEM+"/Graphs/TSNE_Categories_Year0.png")
 plt.show()