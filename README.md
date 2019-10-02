# Volume Stream Prediction 
---
### Code Coverage: ~85%


### Setup:

### Step 1:

#####  | Dependencies:

   1. Create a pip / conda Python 3.7 environment.
   2. Navigate into the project tree.
   3. Run the following command:
    
    # Unix
    sudo pip install -r requirements.txt
    # Windows (Elevated)
    pip install -r requirements.txt


----
### Step 2:

#####  | routes.csv import:

In order for all the tests to be run successfully
 place the routes.csv file that was given in the project description like this:
  
 - module
    - data
      - raw
        - routes.csv
        
----

The prediction functionality can be shown in the respective notebooks for SARIMA and LSTM models,
and follows the same mentality like the one bellow:

    from src.modeling import lstmModel
    from src.processing import dataProc
    
    # Aggregated 1 hour Dataset fetch
    DATASET_PATH = module_path + "/notebook/dt_agg1hour.h5"
    dataset = pd.read_pickle(
        DATASET_PATH)
    
    dataset = dataProc.create_features(dataset= dataset)

    # Model creation
    lstmodel = lstmModel(perform_scale=True)
    
    # Model training
    lstmodel.train(dataset[start_date:end_date], evaluate=False)
    
    # Model usage
    y_pred, y = lstmodel.generate_prediction(input_data= dataset[end_date:pd.Timestamp(end_date)+pd.Timedelta(hours=48)])

    




