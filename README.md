## Introduction

This repository demonstrates how to use incorporate input data validation for machine learning model implementations. We use Pydantic for validating the schema file and the train/test data.

This repository is part of a tutorial series on Ready Tensor, a web platform for AI developers and users. The purpose of the tutorial series is to help AI developers create adaptable algorithm implementations that avoid hard-coding your logic to a specific dataset. This makes it easier to re-use your algorithms with new datasets in the future without requiring any code change.

## Repository Contents

```bash
binary_class_project/
├── examples/
│   ├── titanic_schema.json
│   ├── titanic_train.csv
│   └── titanic_test.csv
├── inputs/
│   ├── data/
│   │   ├── testing/
│   │   └── training/
│   └── schema/
├── model/
│   └── artifacts/
├── outputs/
│   ├── hpt_outputs/
│   ├── logs/
│   └── predictions/
├── src/
│   ├── config/
│   │   ├── default_hyperparameters.json
│   │   ├── hpt.json
│   │   ├── model_config.json
│   │   ├── paths.py
│   │   └── preprocessing.json
│   ├── data_models/
│   │   ├── data_validator.py
│   │   ├── infer_request_model.py
│   │   └── schema_validator.py
│   ├── hyperparameter_tuning/
│   │   ├── __init__.json
│   │   └── tuner.py
│   ├── prediction/
│   │   ├── __init__.json
│   │   └── predictor_model.py
│   ├── preprocessing/
│   │   ├── custom_transformers.py
│   │   ├── pipeline.py
│   │   ├── preprocess.py
│   │   └── target_encoder.py
│   ├── schema/
│   │   └── data_schema.py
│   ├── xai/
│   │   ├── __init__.json
│   │   └── explainer.py
│   ├── logger.py
│   ├── predict.py
│   ├── serve.py
│   ├── serve_utils.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── <mirrors `/src` structure ...>
│   ...
│   ...
│   └── test_utils.py
├── tmp/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

- **`/examples`**: This directory contains example files for the titanic dataset. Three files are included: `titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`. You can place these files in the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- **`/inputs`**: This directory contains all the input files for your project, including the data and schema files. The data is further divided into testing and training subsets.
- **`/model/artifacts`**: This directory is used to store the model artifacts, such as trained models and their parameters.
- **`/outputs`**: The outputs directory contains all output files, including the prediction results, logs, and hyperparameter tuning outputs.
- **`/src`**: This directory holds the source code for the project. It is further divided into various subdirectories such as `config` for configuration files, `data_models` for data models for input validation, `hyperparameter_tuning` for hyperparameter-tuning (HPT) related files, `prediction` for prediction model scripts, `preprocessing` for data preprocessing scripts, `schema` for schema scripts, and `xai` for explainable AI scripts.
- **`/src/data_models`**: This directory contains the data models for input validation. It is further divided into `data_validator.py` for data validation, `infer_request_model.py` for inference request validation, and `schema_validator.py` for schema validation.
- Within **`/src`** folder: We have the following main scripts:
  - **`logger.py`**: This script contains the centralized logger setup for the project. It is used the `train.py`, `predict.py` and `serve.py` scripts. Logging is stored in the path `./app/outputs/logs/`.
  - **`predict.py`**: This script is used to run batch predictions using the trained model. It loads the artifacts and creates and saves the predictions in a file called `predictions.csv` in the path `./app/outputs/predictions/`.
  - **`serve.py`**: This script is used to serve the model as a REST API. It loads the artifacts and creates a FastAPI server to serve the model.
  - **`serve_utils.py`**: This script contains utility functions used by the `serve.py` script.
  - **`train.py`**: This script is used to train the model. It loads the data, preprocesses it, trains the model, and saves the artifacts in the path `./app/outputs/artifacts/`. It also saves a SHAP explainer object in the path `./app/outputs/artifacts/`.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`/tests`**: This directory contains all the tests for the project. It mirrors the `src` directory structure for consistency. There is also a `test_resources` folder inside `/tests` which can contain any resources needed for the tests (e.g. sample data files).
- **`/tmp`**: This directory is used for storing temporary files which are not necessary to commit to the repository.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`LICENSE`**: This file contains the license for the project.
- **`README.md`**: This file contains the documentation for the project, explaining how to set it up and use it.
- **`requirements.txt`**: This file lists the dependencies for the project, making it easier to install all necessary packages.

## Usage

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Place the following 3 input files in the sub-directories in `./app/inputs/`:
  - Train data, which must be a CSV file, to be placed in `./app/inputs/data/training/`. File name can be any; extension must be ".csv".
  - Test data, which must be a CSV file, to be placed in `./app/inputs/data/testing/`. File name can be any; extension must be ".csv".
  - The schema file in JSON format , to be placed in `./app/inputs/data_config/`. The schema conforms to Ready Tensor specification for the **Binary Classification-Base** category. File name can be any; extension must be ".json".
- Run the script `train.py` to train the random forest classifier model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./app/outputs/artifacts/`.
- Run the script `predict.py` to run batch predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./app/outputs/predictions/`.
- Run the script `serve.py` to start the inference service, which can be queried using the `/ping` and `/infer` endpoints. The service also provides local explanations for the predictions using the `/explain` endpoint.

## Validations performed on schema file

The implementation performs the following validations on the schema file:

- **problemCategory** must be set to "binary_classification_base"
- **version** must be set to "1.0"
- **inputDatasets** must not be empty
- **inputDatasets** must have a single key named "binaryClassificationBaseMainInput"
- **idField** must be specified
- **targetField** must be specified
- **targetClass** must be specified
- **predictorFields** must be a non-empty list of objects
- Each **predictorFields** object must contain a **fieldName**
- Each **predictorFields** object must have a valid **dataType** of "**CATEGORICAL**", "**INT**", "**REAL**", or - "**NUMERIC**"

## Validations performed on train and test data files

The implementation performs the following validations on the schema file:

- ID field must be present. The name of the ID field is defined in the schema file.
- Target field must be present if data is for training. The name of the target field is defined in the schema file.
- All categorical or numerical features specified in the schema file must be present in the data file.

## Validations performed on inference request data

The implementation performs the following validations on the inference request data:

- The request body contains a key 'instances' with a list of dictionaries as its value.
- The list is not empty (i.e., at least one instance must be provided).
- Each instance contains the 'id' field whose name is defined in the schema file.
- Each instance contains all the required numerical and categorical features as defined in the schema file.
- Values for each feature in each instance are of the correct data type. Values are allowed to be null (i.e., missing).
- For categorical features, the given value must be one of the categories as defined in the schema file.

## Requirements

The code requires Python 3 and the following libraries:

```makefile
fastapi==0.70.0
uvicorn==0.15.0
pydantic==1.8.2
pandas==1.5.2
numpy==1.20.3
scikit-learn==1.0
feature-engine==1.2.0
imbalanced-learn==0.8.1
scikit-optimize==0.9.0
httpx==0.24.0
shap==0.41.0
```

These packages can be installed by running the following command:

```python
pip install -r requirements.txt
```
