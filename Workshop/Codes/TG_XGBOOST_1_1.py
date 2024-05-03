# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:03:12 2024

@author: agentimis1
"""
#%%
# Basic packages
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
# Preprocessing packages
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
# Model packages
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
# Accuracy Packages
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
#%% 
dir1=os.path.realpath(__file__)
main_dir = os.path.dirname(os.path.dirname(dir1))
#%%
Ex0= pd.read_csv(main_dir+'./Data/FullData.csv')
Ex0.head(5)
#%%
df1=Ex0.describe()
df1
#%%
Ex1=pd.get_dummies(Ex0)
imputer = SimpleImputer(strategy='mean') # Inputs the mean as the missing value
Ex2=pd.DataFrame(imputer.fit_transform(Ex1), columns=Ex1.columns) # applies the imputer to our dataset
#%%
cols = list(Ex2)
cols.insert(0, cols.pop(cols.index('YIELD_OBS')))
Ex2=Ex2.loc[:,cols] # final clean dataset with no missing values and the response variable as the first column
#%%
X=Ex2.iloc[:,1:len(Ex2.columns)].values # makes all the values of the input variables as a matrix
y=Ex2.iloc[:,0].values.flatten() # extracts the y-values
#%%
rf_param_grid = {
    'n_estimators': range(100, 900, 200),
    'max_depth': range(1, 40, 10),
    'max_features': range(1, 60, 10),
}
#%%
xg_param_grid = {
    'learning_rate': np.arange(0, 0.2, 0.05),
    'max_depth': range(1, 10, 2),
    'subsample': np.arange(0.2, 0.6, 0.1),
    'gamma': np.arange(0, 0.4, 0.1)
    }
#%%
rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5, n_jobs=-1) # njobs = -1 is for parallel processing
rf_grid_search.fit(X, y)

xg_grid_search = GridSearchCV(xgb.XGBRegressor(), xg_param_grid, cv=5, n_jobs=-1)
xg_grid_search.fit(X, y)
#%%
rf_best_params = rf_grid_search.best_params_
xg_best_params = xg_grid_search.best_params_
print(rf_best_params)
print(xg_best_params)
#%%
best_rf_model = None
best_rf_mse = 10

best_lr_model = None
best_lr_mse = 10

best_xgb_model = None
best_xgb_mse = 10
#%%
rf_mse_list= []
lr_mse_list=[]
xgb_mse_list = []
reps=50
#%%
for i in range(reps):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Random forest regressor with best parameters
    rf = RandomForestRegressor(**rf_best_params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, y_pred)
    rf_mse_list.append(rf_mse) # adds current mse to the list with each iteration
    if rf_mse < best_rf_mse:
        best_rf_mse = rf_mse
        best_rf_model = rf

    # Linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_y_pred = lr.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_y_pred)
    lr_mse_list.append(lr_mse)
    if lr_mse < best_lr_mse:
        best_lr_mse = lr_mse
        best_lr_model = lr

    # XG Boost
    xgb_regressor = xgb.XGBRegressor(**xg_best_params)
    xgb_regressor.fit(X_train, y_train)
    xgb_y_pred = xgb_regressor.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_y_pred)
    xgb_mse_list.append(xgb_mse)
    if xgb_mse < best_xgb_mse:
        best_xgb_mse=xgb_mse
        best_xgb_model = xgb_regressor

    print("Finished Iteration",i)
#%%
Results=pd.DataFrame({'Random_Forest':rf_mse_list,"Regression":lr_mse_list,'XGBoost':xgb_mse_list})
Results
#%% Computes and Reports mean mse
mean_rf_mse = sum(rf_mse_list) / len(rf_mse_list)
mean_lr_mse = sum(lr_mse_list) / len(lr_mse_list)
mean_xgb_mse = sum(xgb_mse_list) / len(xgb_mse_list)

print("Mean Random Forest mse:", mean_rf_mse)
print("Mean Linear Regression mse:", mean_lr_mse)
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
plt.show()
#%% Calculate predictions for each best model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
RF_pred=best_rf_model.predict(X_test)
LR_pred=best_lr_model.predict(X_test)
XG_pred=best_xgb_model.predict(X_test)
#%% Calculate R2, RMSE, and MAE for each model
r2_RF = r2_score(y_test, RF_pred)
rmse_RF = np.sqrt(mean_squared_error(y_test, RF_pred, squared=False))
mae_RF = mean_absolute_error(y_test, RF_pred)

r2_LR = r2_score(y_test, LR_pred)
rmse_LR = np.sqrt(mean_squared_error(y_test, LR_pred, squared=False))
mae_LR = mean_absolute_error(y_test, LR_pred)

r2_XG = r2_score(y_test, XG_pred)
rmse_XG = np.sqrt(mean_squared_error(y_test, XG_pred, squared=False))
mae_XG= mean_absolute_error(y_test, XG_pred)

mean_y_test = np.mean(y_test)
color_values = (y_test - np.min(y_test)) / (np.max(y_test) - np.min(y_test))

#%% # Plot scatter plots for each model
plt.figure(figsize=(16, 4))

# Random Forest
plt.subplot(1, 3, 1)
plt.scatter(y_test, RF_pred, c=color_values, cmap='viridis', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Measured Yield')
plt.ylabel('Predicted Yield (Random Forest)')
plt.title('Measured vs Predicted Yield (Random Forest)')
plt.text(0.05, 0.95, f'R2: {r2_RF:.2f}\nRMSE: {rmse_RF:.2f}\nMean Y: {mean_y_test:.2f}\nMAE: {mae_RF:.2f}',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()

# XGBOOST
plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 3)
plt.scatter(y_test, XG_pred, c=color_values, cmap='viridis', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Measured Yield')
plt.ylabel('Predicted Yield (XGBOOST)')
plt.title('Measured vs Predicted Yield (XGBOOST)')
plt.text(0.05, 0.95, f'R2: {r2_XG:.2f}\nRMSE: {rmse_XG:.2f}\nMean Y: {mean_y_test:.2f}\nMAE: {mae_XG:.2f}',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()