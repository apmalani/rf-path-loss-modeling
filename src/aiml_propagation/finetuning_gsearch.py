import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('src/aiml_propagation/combined_output_data.csv')

x = data.drop('gamma_ideal', axis = 1)
y = data['gamma_ideal']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'max_features': [1.0, 'sqrt', 0.5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(n_jobs = -1)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose =1)
grid_search.fit(x_train, y_train)

best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print('best params: ', best_params)
print('best score: ', best_score)