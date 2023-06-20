#!/bin/bash

# Import colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Define .env file path
ENV_FILE=".env"

# Set defaults
default_uid=$(id -u)
default_proj_dir="."
default_username="airflow"
default_password="airflow"
default_pip_requirements=""
default_mlflow_backend_store_uri="sqlite:///mlflow.db"
default_mlflow_default_artifact_root="mlruns"
default_image_name="apache/airflow:2.6.1"

# Function to prompt for user input
prompt_input() {
  read -p "$1 [default: $2]: " var
  echo ${var:-$2}
}

# Prompt for user input
echo -e "${BLUE}[+] Configuring the environment for Apache Airflow...${NC}"
echo ""
AIRFLOW_UID=$(prompt_input "Enter AIRFLOW_UID" $default_uid)
AIRFLOW_PROJ_DIR=$(prompt_input "Enter AIRFLOW_PROJ_DIR" $default_proj_dir)
_AIRFLOW_WWW_USER_USERNAME=$(prompt_input "Enter _AIRFLOW_WWW_USER_USERNAME" $default_username)
_AIRFLOW_WWW_USER_PASSWORD=$(prompt_input "Enter _AIRFLOW_WWW_USER_PASSWORD" $default_password)
_PIP_ADDITIONAL_REQUIREMENTS=$(prompt_input "Enter _PIP_ADDITIONAL_REQUIREMENTS" $default_pip_requirements)
MLFLOW_BACKEND_STORE_URI=$(prompt_input "Enter MLFLOW_BACKEND_STORE_URI" $default_mlflow_backend_store_uri)
MLFLOW_DEFAULT_ARTIFACT_ROOT=$(prompt_input "Enter MLFLOW_DEFAULT_ARTIFACT_ROOT" $default_mlflow_default_artifact_root)

echo ""
echo -e "${BLUE}[+] Pay attention to the following Kaggle credentials. If you don't have a Kaggle account, please create one at https://www.kaggle.com/${NC}"
echo ""
KAGGLE_USERNAME=$(prompt_input "Enter KAGGLE_USERNAME" "")
KAGGLE_KEY=$(prompt_input "Enter KAGGLE_KEY" "")

echo ""
echo -e "${BLUE}[+] Pay attention to the following Docker image name.${NC}"
echo ""
AIRFLOW_IMAGE_NAME=$(prompt_input "Enter AIRFLOW_IMAGE_NAME" $default_image_name)

# Create Kaggle directory and JSON file for API credentials
mkdir -p .kaggle
echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" > .kaggle/kaggle.json
chmod 600 .kaggle/kaggle.json

# If .env file exists then make a backup of it
if [ -f "$ENV_FILE" ]; then 
    echo ""
    echo -e "${BLUE}[+] Making a backup of the existing .env file...${NC}"
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

echo ""
echo -e "${GREEN}[+] File $ENV_FILE created/updated with success${NC}"

# Build Docker image with kaggle credentials
echo ""
echo -e "${BLUE}[+] Building Docker image $AIRFLOW_IMAGE_NAME...${NC}"
echo ""
docker build -t ${AIRFLOW_IMAGE_NAME} .

# Check if Docker image was built with success
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[+] Docker image $AIRFLOW_IMAGE_NAME built with success${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}[-] Docker image $AIRFLOW_IMAGE_NAME failed to build${NC}"
    exit 1
fi

# Let's start docker compose stack if everything not failed
read -p "Do you want to start the Docker-compose stack now? [y/N] " confirm

case "$confirm" in 
  [yY][eE][sS]|[yY])
    echo ""
    echo -e "${GREEN}[+] Starting Docker-compose stack...${NC}"
    echo ""
    docker-compose up -d
    echo ""
    echo -e "${GREEN}[+] Docker-compose stack is up and running!${NC}"
    echo ""
    ;;
  *)
    echo ""
    echo -e "${RED}[-] Docker-compose stack was not started${NC}"
    echo ""
    ;;
esac