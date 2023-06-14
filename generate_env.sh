#!/bin/bash

# Define .env file path
ENV_FILE=".env"

# Prompt for user input
read -p "Enter AIRFLOW_UID (default is current user id $(id -u)): " input_uid
read -p "Enter AIRFLOW_IMAGE_NAME (default is apache/airflow:2.6.1): " input_image_name
read -p "Enter AIRFLOW_PROJ_DIR (default is '.'): " input_proj_dir
read -p "Enter _AIRFLOW_WWW_USER_USERNAME (default is 'airflow'): " input_username
read -p "Enter _AIRFLOW_WWW_USER_PASSWORD (default is 'airflow'): " input_password
read -p "Enter _PIP_ADDITIONAL_REQUIREMENTS (default is ''): " input_pip_requirements
read -p "Enter MLFLOW_BACKEND_STORE_URI (default is 'sqlite:///mlflow.db'): " input_mlflow_backend_store_uri
read -p "Enter MLFLOW_ARTIFACT_ROOT (default is 'mlruns'): " input_mlflow_default_artifact_root

# Check if user input is not empty else use default values
AIRFLOW_UID=${input_uid:=$(id -u)}
AIRFLOW_IMAGE_NAME=${input_image_name:="apache/airflow:2.6.1"}
AIRFLOW_PROJ_DIR=${input_proj_dir:="."}
_AIRFLOW_WWW_USER_USERNAME=${input_username:="airflow"}
_AIRFLOW_WWW_USER_PASSWORD=${input_password:="airflow"}
_PIP_ADDITIONAL_REQUIREMENTS=${input_pip_requirements:=""}

MLFLOW_BACKEND_STORE_URI=${input_mlflow_backend_store_uri:="sqlite:///mlflow.db"}
MLFLOW_DEFAULT_ARTIFACT_ROOT=${input_mlflow_default_artifact_root:="mlruns"}

# If .env file exists then make a backup of it
if [ -f "$ENV_FILE" ]; then 
    echo "Making a backup of the existing .env file..."
    cp "$ENV_FILE" "$ENV_FILE.bak"
fi

# Write to the .env file
cat > "$ENV_FILE" << EOF
AIRFLOW_UID=$AIRFLOW_UID
AIRFLOW_IMAGE_NAME=$AIRFLOW_IMAGE_NAME
AIRFLOW_PROJ_DIR=$AIRFLOW_PROJ_DIR
_AIRFLOW_WWW_USER_USERNAME=$_AIRFLOW_WWW_USER_USERNAME
_AIRFLOW_WWW_USER_PASSWORD=$_AIRFLOW_WWW_USER_PASSWORD
_PIP_ADDITIONAL_REQUIREMENTS=$_PIP_ADDITIONAL_REQUIREMENTS
MLFLOW_BACKEND_STORE_URI=$MLFLOW_BACKEND_STORE_URI
MLFLOW_DEFAULT_ARTIFACT_ROOT=$MLFLOW_DEFAULT_ARTIFACT_ROOT
EOF

echo "File $ENV_FILE created/updated with success"