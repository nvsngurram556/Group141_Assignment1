# Group141_Assignment1
This repo is mainly used to submit the MLOps Assignment which was assigned to Group 141
Groups members are:
1. Gurram Naga Venkata Satya Narayana - 2023ac05359
2. Arunarkavalli Darbha - 2023ac05295
3. Rishikesh Ashokrao Bakale - 2023ac05046
4. Aditya Gupta - 2023ac05742

Directory structure:
Group141_Assignment1/
├── .github/
│   └── workflows/
├── data/
│   ├── raw/           # DVC-tracked raw files (tracked via dvc add)
│   └── processed/     # processed files (optionally tracked)
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── preprocess.py
│   ├── models/
│   │   └── train.py
│   └── api/
│       └── app.py
├── notebooks/
├── Dockerfile
├── requirements.txt
├── dvc.yaml           # optional pipeline stages
├── .gitignore
└── README.md

created AWS S3 bucket and user for pushing data versions
AWS user : dvc-user
AWS s3 bucket : aws-california-housing-dvc  

# Part 1: execution instructions:
git init
pip install -r requirements.txt

dvc init
dvc remote add -d s3remote s3://aws-california-housing-dvc/California_Housing
dvc remote modify s3remote profile dvc-user

python src/data/make_dataset.py

dvc add data/raw/california_housing_raw.csv
git add data/raw/california_housing_raw.csv.dvc .gitignore
git commit -m "add raw data"

dvc push

# Part 2: Model Development & Experiment Tracking
# optional: set MLflow tracking uri to local server or leave default (mlruns)
export MLFLOW_TRACKING_URI=file:./mlruns
python src/models/train_model.py --data data/processed/california_housing_processed.csv --experiment-name california-housing --dt-max-depth 10

mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns -p 5000
# then open http://127.0.0.1:5000

python src/models/select_and_register_model.py

