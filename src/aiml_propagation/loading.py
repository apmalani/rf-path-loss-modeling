import joblib
from sklearn.metrics import mean_squared_error
import pandas as pd

data = pd.read_csv('src/aiml_propagation/combined_output_data.csv')

x = data.drop(['gamma_ideal', 'city'], axis=1)
y = data['gamma_ideal']
city_info = data['city'] 

model = joblib.load('results/trained_final_model.pkl')

y_pred = model.predict(x)

diff = abs(y - y_pred)

results = pd.DataFrame({
    'actual': y,
    'predicted': y_pred,
    'difference': diff,
    'city': city_info 
})

results.to_csv('results/running_loaded_predictions.csv', index=False)

mse = mean_squared_error(y, y_pred)
print("MSE: ", mse)

rmse = mse ** 0.5
print('RMSE: ', rmse)
