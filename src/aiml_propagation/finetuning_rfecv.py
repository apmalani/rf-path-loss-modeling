import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

data = pd.read_csv('src/aiml_propagation/combined_output_data.csv')

x = data.drop('gamma_ideal', axis = 1)
y = data['gamma_ideal']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

rf = RandomForestRegressor(n_jobs = -1)

rfecv = RFECV(rf, cv = 5)

rfecv.fit(x_train, y_train)

print(f'opt. # of features: {rfecv.n_features_}') # can change with randomness, but in general lower the better

features = rfecv.support_
print(f'features: {features}') # we ended up with 13, see preprocess.py