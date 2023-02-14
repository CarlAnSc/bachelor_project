FROM python:3.9-slim

# Update and upgrade the dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*
    
# Install requirements
COPY requirements_app.txt requirements_app.txt
RUN pip install --upgrade pip
RUN pip install -r requirements_app.txt --no-cache-dir

# copy app
COPY App App/
COPY Data_Analysis/imgs_for_app.csv Data_Analysis/imgs_for_app.csv
COPY Data_Analysis/imgs_for_intro.csv Data_Analysis/imgs_for_intro.csv

# Install source as package
#---COPY setup.py setup.py
#---RUN pip install -e .
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run train
ENTRYPOINT ["streamlit", "run", "App/Introduction.py", "--server.port=8501", "--server.address=0.0.0.0"]
