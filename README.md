# aiml_propagation
### Arun Malani, Matthew Pugh, Jonathan Lu, Chrysanthos Chrysanthou

This package contains scripts and routines for AI/ML-based propagation prediction.

## Set-up
1. ```git clone```  the repo
2. Move all pertinent data into the correct folder

## Usage
1. Loading
    - Preprocess your new data by using preprocess.py
        - Change the "dir" variable to the directory of the data you want to load
    - Access the model
        - Quickly run model.py to generate the .pkl file (a few minutes max)
        - OR load it from drive (slower)
    - Run loading.py
        - Access the output data via the numpy.ndarray or the resulting CSV file
2. Training
    - Preprocess your new data by using preprocess.py
        - Do NOT change the "dir" variable, just add your new data to the training set
    - Run model.py
        - If you believe that the added data will substantially change the model's weights (VERY unlikely unless the new data is very big):
            - Rerun finetuning_rfecv.py to generate the new optimal features
            - Rerun finetuning_gsearch.py to generate the new optimal parameters
            - Rerun preprocess.py, changing the "rmv" variable to include or exclude the features you generated
            - Rerun model.py, changing the "params" variable to match the paramters you generated
3. Other
    - Run finetuning_factoring.py to observe a bar chart of the top weighted features
    - The file "results/training_predicts_v_actuals.csv" displays the results and differences for every observation in the dataset
        - Currently does so in regards to the input "combined_output_data.csv"
    - The file "results/running_loaded_predictions.csv" displays the pure predictions for the loaded observations 
        - Currently does so in regards to ALL of the input data, including data it has already seen 
        - **This is the file that contains all future predictions based on the given pertinent data**


