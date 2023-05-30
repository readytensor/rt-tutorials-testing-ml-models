## Introduction

This repository is an implementation of the random forest model for the binary classification task. This repository is part of a tutorial series on Ready Tensor, a web platform for AI developers and users. The purpose of the tutorial series is to help AI developers create adaptable algorithm implementations that avoid hard-coding your logic to a specific dataset. This makes it easier to re-use your algorithms with new datasets in the future without requiring any code change.

This particular repository is referenced in the tutorial [Testing ML models] on the Ready Tensor website. This tutorial covers the following topics:

- Implementing Unit Tests: Focusing on writing unit tests for our binary classifier model using pytest, a popular testing framework.
- Implementing Integration Tests: Covering integration tests using pytest for our binary classifier model.
- Coverage Testing: Exploring code coverage analysis during development using pytest-cov, a powerful tool to measure code coverage and ensure a significant portion of your codebase is exercised during the testing process.
- Performance Testing: Evaluating the efficiency and scalability of ML models during development under various conditions, helping identify potential bottlenecks and ensuring that your model meets desired performance criteria.

We have included all four types of tests: unit, integration, coverage and performance in the repository.

Note that the overall implementation contains:

- A data schema definition created as per Ready Tensor specifications for the binary classifier problem.
- A flexible preprocessing pipeline that can be easily adapted to new datasets, and other classifier models. We use SciKit-Learn and feature-engine to implement the preprocessing pipeline.
- A random forest classifier built using Scikit-Learn.
- A SHAP explainer built using the SHAP library. This explainer is used to provide local explanations for the predictions made by the model.
- Hyperparameter tuning for the Random Forest model's hyperparameters built using Scikit-Optimize.
- A FastAPI server to serve the model as a REST API.
- Input data validation using Pydantic. This includes validation on the schema file, train/prediction input files, and inference request body.

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
│   ├── integration_tests/
│   ├── performance_tests/
│   ├── test_resources/
│   ├── test_results/
│   ├── unit_tests/
│   │   ├── <mirrors /src structure>
│   │   └── ...
│   ├── __init__.py
│   └── conftest.py
├── tmp/
├── .gitignore
├── LICENSE
├── pytest.ini
├── README.md
├── requirements.txt
└── requirements-test.txt
```

- **`/examples`**: This directory contains example files for the titanic dataset. Three files are included: `titanic_schema.json`, `titanic_train.csv` and `titanic_test.csv`. You can place these files in the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- **`/inputs`**: This directory contains all the input files for your project, including the data and schema files. The data is further divided into testing and training subsets.
- **`/model/artifacts`**: This directory is used to store the model artifacts, such as trained models and their parameters.
- **`/outputs`**: The outputs directory contains all output files, including the prediction results, logs, and hyperparameter tuning outputs.
- **`/src`**: This directory holds the source code for the project. It is further divided into various subdirectories such as `config` for configuration files, `data_models` for data models for input validation, `hyperparameter_tuning` for hyperparameter-tuning (HPT) related files, `prediction` for prediction model scripts, `preprocessing` for data preprocessing scripts, `schema` for schema scripts, and `xai` for explainable AI scripts.
- **`/src/data_models`**: This directory contains the data models for input validation. It is further divided into `data_validator.py` for data validation, `infer_request_model.py` for inference request validation, and `schema_validator.py` for schema validation.
- Within **`/src`** folder: We have the following main scripts:
  - **`logger.py`**: This script contains the centralized logger setup for the project. It is used the `train.py`, `predict.py` and `serve.py` scripts. Logging is stored in the path `./outputs/logs/`.
  - **`predict.py`**: This script is used to run batch predictions using the trained model. It loads the artifacts and creates and saves the predictions in a file called `predictions.csv` in the path `./outputs/predictions/`.
  - **`serve.py`**: This script is used to serve the model as a REST API. It loads the artifacts and creates a FastAPI server to serve the model.
  - **`serve_utils.py`**: This script contains utility functions used by the `serve.py` script.
  - **`train.py`**: This script is used to train the model. It loads the data, preprocesses it, trains the model, and saves the artifacts in the path `./model/artifacts/`. It also saves a SHAP explainer object in the path `./model/artifacts/`. When the train task is run with a flag to perform hyperparameter tuning, it also saves the hyperparameter tuning results in the path `./outputs/hpt_outputs/`.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`/tests`**: This directory contains all the tests for the project and associated resources and results.
  - **`integration_tests.py`**: This directory contains all the integration tests for the project. We cover four main workflows: data preprocessing, training, prediction, and inference service.
  - **`performance_tests.py`**: This directory contains performance tests for the training and batch prediction workflows in the script `test_train_predict.py`. It also contains performance tests for the inference service workflow in the script `test_inference_apis.py`. Helper functions are defined in the script `performance_test_helpers.py`. Fixtures and other setup are contained in the script `conftest.py`.
  - **`test_resources.py`**: This folder contains various resources needed in the tests, such as trained model artifacts (including the preprocessing pipeline, target encoder, explainer, etc.). These resources are used in integration tests and performance tests.
  - **`test_results.py`**: This folder contains the results for the performance tests. These are persisted to disk for later analysis.
  - **`unit_tests.py`**: This folder contains all the unit tests for the project. It is further divided into subdirectories mirroring the structure of the `src` folder. Each subdirectory contains unit tests for the corresponding script in the `src` folder.
- **`/tmp`**: This directory is used for storing temporary files which are not to be committed to the repository.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`LICENSE`**: This file contains the license for the project.
- **`pytest.ini`**: This file contains the configuration for pytest, including the markers used for tests.
- **`README.md`**: This file contains the documentation for the project, explaining how to set it up and use it.
- **`requirements.txt`**: This file lists the dependencies for the project, particularly to run the scripts in the `src` folder.
- **`requirements-test.txt`**: This file lists the testing dependencies for the project. These are needed to run the tests in the `tests` folder.

## Usage

To run the project:

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Place the following 3 input files in the sub-directories in `./src/inputs/`:
  - Train data, which must be a CSV file, to be placed in `./src/inputs/data/training/`. File name can be any; extension must be ".csv".
  - Test data, which must be a CSV file, to be placed in `./src/inputs/data/testing/`. File name can be any; extension must be ".csv".
  - The schema file in JSON format , to be placed in `./src/inputs/data_config/`. The schema conforms to Ready Tensor specification for the **Binary Classification-Base** category. File name can be any; extension must be ".json".
- Run the script `train.py` to train the random forest classifier model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./model/artifacts/`.
- Run the script `predict.py` to run batch predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./outputs/predictions/`.
- Run the script `serve.py` to start the inference service, which can be queried using the `/ping` and `/infer` endpoints. The service also provides local explanations for the predictions using the `/explain` endpoint.

To run the tests:

- Install dependencies listed in `requirements-test.txt`.
- Run the command `pytest` from the root directory of the repository.
- To run specific scripts, use the command `pytest <path_to_script>`.
- To run performance tests (which take longer to run): use the command `pytest -m performance`.

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

For testing, the following packages are required:

```makefile
pytest==6.2.5
pytest-cov==3.0.0
```

You can install these packages by running the following command:

```python
pip install -r requirements-test.txt
```
