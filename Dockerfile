# Start with the official Apache Airflow image
FROM apache/airflow:2.6.1-python3.9

# Switch back to the home directory
WORKDIR /home/${APP_USER}

# Copy the requirements file into the image
COPY requirements.txt .

# Install the Python requirements
RUN pip install --no-cache-dir -r requirements.txt