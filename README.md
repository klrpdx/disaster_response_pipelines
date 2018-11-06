# Disaster Response Pipeline Project

A machine learning pipeline that injests social media messages related to natural disasters and outputs a model trained 
to classify these messages according to subject, urgency, and need. The trained model is hosted via a Flask web app that
allows the user to enter messages and see how they are classified.

### What's inside:
-  Example data files:
    -  disaster_messages.csv: Archive of messages from natural disasters
    -  disater_categories.csv: Categories the messages can be mapped to
    
-  Python scripts:
    -  process_data.py: Creates a Pandas dataframe from the messages and categories files; cleans and processes text;
    stores the dataframe in a sqlite database table
    -  train_classifier.py: Reads dataframe from sqlite, trains a model, and evaluates the model's performance. Stores
    the model in a pickle file.
    - run.py: Runs the entire ML pipeline process and runs a web app for classification of messages. See instructions
    below. 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
