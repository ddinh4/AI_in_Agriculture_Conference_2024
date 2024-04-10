# -*- coding: utf-8 -*-
"""
Created on 02/29/2024

@author: agentimis1
"""
#%%
# Basic packages
import pandas as pd  
import os
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
#import joblib

# Preprocessing packages
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import  LabelEncoder
from sklearn.impute import SimpleImputer

# Model packages
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor

# mse Packages

from sklearn.metrics import mean_squared_error



#=== Unused packages
#from sklearn.metrics import classification_report
#from keras import models, layers
#from tensorflow.keras.layers import Dropout

#%% Sets the working directory to the main directory containing the project
#dir1=os.path.realpath()
#main = os.path.dirname(os.path.dirname(dir1))
#%% Reads the dataframe
Ex0= pd.read_csv('G:/.shortcut-targets-by-id/1N2GeQNhCJy4B6-unK1KaiWfePgNe9k29//Dina_Dinh/2024_Spring_AIConference/Workshop/Data/HousingData.csv',low_memory=False)
#%% Imputes, using most frequent and creates the Ex2 dataframe
imputer = SimpleImputer(strategy='mean') # Inputs the mean as the missing value
Ex1=pd.DataFrame(imputer.fit_transform(Ex0), columns=Ex0.columns) # applies the imputer to our dataset
cols = list(Ex1)
cols.insert(0, cols.pop(cols.index('MEDV')))
Ex2=Ex1.loc[:,cols] # final clean dataset with no missing values and the response variable as the first column
#%% Splits to input and output
X=Ex2.iloc[:,1:len(Ex2.columns)].values # makes all the values of the input variables as a matrix
y=Ex2.iloc[:,0].values.flatten() # extracts the y-values

#%% Extracts the number of levels of the output variable (not necessary for regression)
#num_levels = Ex2.iloc[:,0].nunique()
#%% Sets up a parameter estimation grid
param_grid = {
    'n_estimators': range(100, 800, 150),
    'max_depth': range(1, 50, 10),
    'max_features': range(3, 20, 5),
}

# XGBoost optimization help from: https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning &
# https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster
xg_param_grid = {
    'learning_rate': np.arange(0, 0.2, 0.01), # shrinking feature weights to prevent overfitting (slows down process when value is lower which is why it prevents overfitting)
    'max_depth': range(1, 10, 2), # same as RF
    'subsample': np.arange(0.2, 0.6, 0.1), # Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
    'colsample_bytree': np.arange(0.1, 0.5, 0.1),
    'gamma': np.arange(0, 0.4, 0.1) # specifies the minimum loss reduction required to make a split. The larger gamma is, the more conservative 
    } # no regularization done, but can tune hyperparameters for regularizing
#%% Performs Grid search using all data
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=-1) # njobs = -1 is for parallel processing
grid_search.fit(X, y)

xg_grid_search = GridSearchCV(xgb.XGBRegressor(), xg_param_grid, cv=5, n_jobs=-1)
xg_grid_search.fit(X, y)
#%% Computes best parameters and the corresponding regressor
best_params = grid_search.best_params_
df1 = pd.DataFrame([best_params])
rf_best_param = os.path.join('G:/.shortcut-targets-by-id/1N2GeQNhCJy4B6-unK1KaiWfePgNe9k29//Dina_Dinh/2024_Spring_AIConference/Workshop/Results/', 'rf_best_param.csv')
df1.to_csv(rf_best_param, index = False)

xg_best_params = xg_grid_search.best_params_
df2 = pd.DataFrame([xg_best_params])
xg_best_param = os.path.join('G:/.shortcut-targets-by-id/1N2GeQNhCJy4B6-unK1KaiWfePgNe9k29//Dina_Dinh/2024_Spring_AIConference/Workshop/Results/', 'xg_best_param.csv')
df2.to_csv(xg_best_param, index = False)

#%% Initialize variables to store best model information
best_rf_model = None
best_rf_mse = 0.0

best_lr_model = None
best_lr_mse = 0.0

#best_nb_model = None
#best_nb_mse = 0.0

best_xgb_model = None
best_xgb_mse = 0.0
#%% Inititalize vectors to store the mse and set the number of repetitions
rf_mse_list= []
lr_mse_list=[]
#nb_mse_list = []
xgb_mse_list = []

reps=50
#%% Scales and encodes the data (Not needed for regression)
#label_encoder = LabelEncoder() # encodes categorical levels to numbers
#y_encoded = label_encoder.fit_transform(y)
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # standardize the X variables

#%% Main Loop
for i in range(reps):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=None)
    
    # Random forest regressor with best parameters
    rf = RandomForestRegressor(**best_params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, y_pred)
    rf_mse_list.append(rf_mse) # adds current mse to the list with each iteration
    if rf_mse < best_rf_mse:
        best_rf_mse = rf_mse
        best_rf_model = rf
    
    # linear regression
    lr = LinearRegression() 
    lr.fit(X_train, y_train)
    lr_y_pred = lr.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_y_pred)
    lr_mse_list.append(lr_mse)
    if lr_mse < best_lr_mse:
        best_lr_mse = lr_mse
        best_lr_model = lr
    
    #Naive Bayes Classifier
    #nb = GaussianNB()
    #nb.fit(X_train, y_train)
    #nb_y_pred = nb.predict(X_test)
    #nb_mse = mean_squared_error(y_test, nb_y_pred)
    #nb_mse_list.append(nb_mse)
    #if nb_mse < best_nb_mse:
      #  best_nb_mse = nb_mse
      #  best_nb_model = nb
        
    # XG Boost
    xgb_regressor = xgb.XGBRegressor(**xg_best_params)
    xgb_regressor.fit(X_train, y_train)
    xgb_y_pred = xgb_regressor.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_y_pred)
    xgb_mse_list.append(xgb_mse)
    if xgb_mse < best_xgb_mse:
        best_xgb_mse=xgb_mse
        best_xgb_model = xgb_regressor     
        
    print("Finished Itteration",i)
#%% Saving best model RF    
#joblib.dump(best_rf_model, 'G:/.shortcut-targets-by-id/1N2GeQNhCJy4B6-unK1KaiWfePgNe9k29//Dina_Dinh/2024_Spring_AIConference/Workshop/Results/rf_best_model.pkl')   
#joblib.dump(best_xgb_model, "best_model_XGBoost.pkl")
#joblib.dump(best_xgb_model, 'G:/.shortcut-targets-by-id/1N2GeQNhCJy4B6-unK1KaiWfePgNe9k29//Dina_Dinh/2024_Spring_AIConference/Workshop/Results/xg_best_model.pkl') 
#%% Compiles all the mse in a dataframe and saves it
Results=pd.DataFrame({'Random_Forest':rf_mse_list,"Regression":lr_mse_list,'XGBoost':xgb_mse_list})
Results.to_csv('G:/.shortcut-targets-by-id/1N2GeQNhCJy4B6-unK1KaiWfePgNe9k29//Dina_Dinh/2024_Spring_AIConference/Workshop/Results/model_comparison_results.csv')
#%% Computes and Reports mean mse
mean_rf_mse = sum(rf_mse_list) / len(rf_mse_list)
mean_lr_mse = sum(lr_mse_list) / len(lr_mse_list)
#mean_nb_mse = sum(nb_mse) / len(nb_mse)
mean_xgb_mse = sum(xgb_mse_list) / len(xgb_mse_list)

print("Mean Random Forest mse:", mean_rf_mse)
print("Mean Linear Regression mse:", mean_lr_mse)
#print("Mean Gaussian Naive Bayes mse:", mean_nb_mse)
print("Mean XGBoost mse:", mean_xgb_mse)
#%% Plotting mse
fig, ax = plt.subplots()
ax.boxplot(Results)
ax.set_title('Side by Side Boxplot of MSE for different Models')
ax.set_xlabel('Predictive Models')
ax.set_ylabel('mse')
xticklabels=['Random Forest','Linear Regression','XGBoost']
ax.set_xticklabels(xticklabels)
ax.yaxis.grid(True)
plt.savefig("G:/.shortcut-targets-by-id/1N2GeQNhCJy4B6-unK1KaiWfePgNe9k29//Dina_Dinh/2024_Spring_AIConference/Workshop/Results/Side_by_Side_Models.png")

#%% Showing and saving the confusion matrix for the best RF
#y_pred = best_rf_model.predict(X_test)
#conf_matrix = confusion_matrix(y_test, y_pred)
#levels = np.unique(y_test)
#conf_matrix_df = pd.DataFrame(conf_matrix, index=levels, columns=levels)
#print("Confusion Matrix:")
#print(conf_matrix_df)
#conf_matrix_df.to_csv(main+'/Results/F01_Majors/Confusion/Conf_Matrix_Majors70_RF.csv')
#%% Computes the most important features in the RF
#feature_importances=best_rf_model.feature_importances_
#most_important_indices = feature_importances.argsort()[:][::-1]  # Change 10 to the desired number of features
#Important_Input=Ex1.iloc[:,most_important_indices]
#most_important_features=Important_Input.columns.tolist()
#%% Plot feature importances for the RF
#plt.figure(figsize=(10, 6))
#plt.barh(range(len(most_important_features)), feature_importances[most_important_indices], align='center')
#plt.yticks(range(len(most_important_features)), most_important_features)
#plt.xlabel('Feature Importance')
#plt.ylabel('Features')
#plt.title('Most Significant Predictors')
#plt.savefig("G:/.shortcut-targets-by-id/1N2GeQNhCJy4B6-unK1KaiWfePgNe9k29//Dina_Dinh/2024_Spring_AIConference/Workshop/Results/FeatureImportanceRF.png")

#%% Perform t-SNE for dimensionality reduction with 2 components
#tsne = TSNE(n_components=2, random_state=42)
#reduced_data = tsne.fit_transform(X_scaled)

#%% Combine reduced data with target variable
#reduced_df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2'])
#reduced_df['Majors'] = y

#%% Plot t-SNE clusters
#plt.figure(figsize=(10, 8))
#sns.scatterplot(x='Component 1', y='Component 2', hue='Majors', data=reduced_df, palette='tab10', legend='full')
#plt.title('t-SNE Visualization of Majors')
#plt.xlabel('Component 1')
#plt.ylabel('Component 2')
#plt.savefig(main+"/Results/F01_Majors/Graphs/TSNE_Majors70.png")
#plt.show()