import ast  # Import the ast module for abstract syntax tree manipulation
import pandas as pd  # Import pandas for data manipulation and analysis
import numpy as np  # Import numpy for numerical computations

from datetime import datetime  # Import datetime module for working with dates and times

from airflow import DAG  # Import the DAG class from Airflow
from airflow.operators.python import PythonOperator  # Import the PythonOperator class from Airflow

# Function to safely evaluate and parse a string containing a literal Python expression
def f(x):
    return ast.literal_eval(x.rstrip('\r\n'))

# Function to calculate bounding box coordinates from a geometry object
def get_bounds(geometry):
    try:
        arr = np.array(geometry).T
        xmin, ymin = np.min(arr[0]), np.min(arr[1])
        xmax, ymax = np.max(arr[0]), np.max(arr[1])
        return (xmin, ymin, xmax, ymax)
    except:
        return np.nan

# Functions to calculate width and height from bounding box coordinates
def get_width(bounds):
    try:
        xmin, _, xmax, _ = bounds
        return np.abs(xmax - xmin)
    except:
        return np.nan

def get_height(bounds):
    try:
        _, ymin, _, ymax = bounds
        return np.abs(ymax - ymin)
    except:
        return np.nan

# Functions to extract x and y coordinates from bounding box
def get_x(bounds):
    try:
        xmin, _, _, _ = bounds
        return np.abs(xmin)
    except:
        return np.nan

def get_y(bounds):
    try:
        _, ymin, _, _ = bounds
        return np.abs(ymin)
    except:
        return np.nan

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
}

# Create a DAG instance with the given parameters
with DAG('aircraft_dataset_download_and_preprocess_dag', default_args=default_args, schedule_interval='@daily', catchup=False) as dag:
    
    # Define a function to download the dataset using Kaggle API
    def download_dataset(dataset, output_path):
        import kaggle  # Import the kaggle module for interacting with Kaggle API
        kaggle.api.authenticate()  # Authenticate with Kaggle API
        kaggle.api.dataset_download_files(dataset, path=output_path, unzip=True)  # Download and unzip the dataset
    
    # Create a PythonOperator to execute the dataset download task
    downloading_dataset = PythonOperator(
        task_id='aircraft_dataset_download',
        python_callable=download_dataset,
        op_kwargs={
            'dataset': 'airbusgeo/airbus-aircrafts-sample-dataset',
            'output_path': '/opt/airflow/data/'
        }
    )

    # Define a function to preprocess the downloaded dataset
    def preprocess_dataset(input_path: str, output_path: str):
        df = pd.read_csv(input_path, converters={'geometry': f, 'class': lambda o: 'airplane'})
        df.loc[:,'bounds'] = df.loc[:,'geometry'].apply(get_bounds)
        df.loc[:,'w'] = df.loc[:,'bounds'].apply(get_width)
        df.loc[:,'h'] = df.loc[:,'bounds'].apply(get_height)
        df.loc[:,'x'] = df.loc[:,'bounds'].apply(get_x)
        df.loc[:,'y'] = df.loc[:,'bounds'].apply(get_y)
        df = df.drop(columns=['geometry', 'bounds'])
        df.to_csv(output_path, index=False)  # Save the modified DataFrame to a CSV file
    
    # Create a PythonOperator to execute the dataset preprocessing task
    preprocessing_dataset = PythonOperator(
        task_id='aircraft_dataset_preprocessing',
        python_callable=preprocess_dataset,
        op_kwargs={
            'input_path': '/opt/airflow/data/annotations.csv',
            'output_path': '/opt/airflow/data/annotations.csv'
        }
    )

    # Set the task dependencies: downloading_dataset must run before preprocessing_dataset
    downloading_dataset >> preprocessing_dataset
