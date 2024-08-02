import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# replace with the CSV of the *NEW* data (currently uses all of the old data)
data = pd.read_csv('src/aiml_propagation/combined_output_data.csv')

x = data.drop('gamma_ideal', axis = 1)
y = data['gamma_ideal']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1, random_state = 42)

model = joblib.load('results/trained_rf_model_grid1.pkl')

# predictions
# this returns a numpy.ndarray with the predicted gamma values - this is the final answer
y_pred = model.predict(x_test)

# evaluating the model 
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
