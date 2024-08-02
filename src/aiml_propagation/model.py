import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# data is already preprocessed (with features removed)
data = pd.read_csv('src/aiml_propagation/combined_output_data.csv')

x = data.drop('gamma_ideal', axis = 1)
y = data['gamma_ideal']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# params obtained from gsearch
params = {
    'max_depth': 30, 'max_features': 0.5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 1000, 'n_jobs' : -1, 'verbose': 1, 'random_state': 42
}

rf = RandomForestRegressor(**params)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

diff = abs(y_test - y_pred)

results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'difference': diff
})

results.to_csv('results/predicts_v_actuals.csv')

joblib.dump(rf, 'results/trained_final_model.pkl')
