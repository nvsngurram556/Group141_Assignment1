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
# Step 1: run the load data python file "python src/data/load_data.py"
# Step 2: run the preprocess file "python src/features/preprocess.py"

# Step 3: run dvc to add file "dvc add data/raw/california_housing_raw.csv"
git add data/raw/california_housing_raw.csv.dvc .gitignore
git commit -m "add raw data"

# Step 4: push the dvc version file to S3 bucket "dvc push"

# Part 2: Model Development & Experiment Tracking
# optional: set MLflow tracking uri to local server or leave default (mlruns)
export MLFLOW_TRACKING_URI=file:./mlruns
# Step 5: python src/models/train_model.py --data data/processed/california_housing_processed.csv --experiment-name california-housing --dt-max-depth 10
# Step 6: python src/models/select_and_register_model.py
# Step 7: mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns -p 5000
# then open http://127.0.0.1:5000

# Part 3: API & Docker Packing
follow below steps for creating and docker packing
# Step 8: python src/api/app.py to run the application using api
# build the docker package with name housing-api
# Step 9: docker build -t housing-api .
# Step 10: docker build --progress=plain --no-cache -t housing-api .

# ake sure mlflow ui is running on http://127.0.0.1:5000 and docker build should be build then follow below run command
# Step 11: docker run -p 8000:8000 \
  -v $(pwd)/mlruns:/app/mlruns \
  -e MODEL_URI="mlruns/518425936383115905/34a7276f614c4a7aa1d18fc6fb047ad9/artifacts/model" \
  housing-api

# now we can run the prediction command by passing input with json and expect the output:
# input:
# Step 12: curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841, "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23, "MedHouseVal": 69.3}'
# output:
{"prediction":4.214499999999999}%    


# Part 4: 

docker build -t nvsngurram/housing-app .
docker run -d -p 8000:8000 nvsngurram/housing-app