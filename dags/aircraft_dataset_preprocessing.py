import ast

import pandas as pd
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def f(x): 
    return ast.literal_eval(x.rstrip('\r\n'))

def get_bounds(geometry):
    try: 
        arr = np.array(geometry).T
        xmin = np.min(arr[0])
        ymin = np.min(arr[1])
        xmax = np.max(arr[0])
        ymax = np.max(arr[1])
        return (xmin, ymin, xmax, ymax)
    except:
        return np.nan

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
    
def preprocess_dataset(input_path: str, output_path: str):
    # Create a DataFrame from the CSV file
    df = pd.read_csv(input_path, converters={'geometry': f, 'class': lambda o: 'airplane'})

    # Create bounds, width and height columns
    df.loc[:,'bounds'] = df.loc[:,'geometry'].apply(get_bounds)
    df.loc[:,'w'] = df.loc[:,'bounds'].apply(get_width)
    df.loc[:,'h'] = df.loc[:,'bounds'].apply(get_height)
    df.loc[:,'x'] = df.loc[:,'bounds'].apply(get_x)
    df.loc[:,'y'] = df.loc[:,'bounds'].apply(get_y)

    # Save the modified dataframe to the same CSV file
    df.to_csv(output_path, index=False)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
}

with DAG('aircraft_dataset_preprocessing_dag', default_args=default_args, schedule='@daily', catchup=False) as dag:
    download_dataset = PythonOperator(
        task_id='aircraft_dataset_preprocessing',
        python_callable=preprocess_dataset,
        op_kwargs={
            'input_path': '/opt/airflow/data/annotations.csv',
            'output_path': '/opt/airflow/data/annotations.csv'
        }
    )