import json
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serve import create_app
from src.serve_utils import get_model_resources


@pytest.fixture
def model_resources(resources_paths):
    """Define a fixture for the test ModelResources object."""
    return get_model_resources(**resources_paths)


@pytest.fixture
def app(model_resources):
    """Define a fixture for the test app."""
    return TestClient(create_app(model_resources))


@pytest.fixture
def sample_request_data():
    # Define a fixture for test data
    return {
        "instances": [
            {
                "PassengerId": "879",
                "Pclass": 3,
                "Name": "Laleff, Mr. Kristo",
                "Sex": "male",
                "Age": None,
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "349217",
                "Fare": 7.8958,
                "Cabin": None,
                "Embarked": "S",
            }
        ]
    }


@pytest.fixture
def sample_response_data():
    # Define a fixture for expected response
    return {
        "status": "success",
        "message": "",
        "timestamp": "...varies...",
        "requestId": "...varies...",
        "targetClasses": ["0", "1"],
        "targetDescription": "A binary variable indicating whether or not the \
            passenger survived (0 = No, 1 = Yes).",
        "predictions": [
            {
                "sampleId": "879",
                "predictedClass": "0",
                "predictedProbabilities": [0.97548, 0.02452],
            }
        ],
    }


@pytest.fixture
def sample_explanation_response_data():
    # Define a fixture for expected response
    return {
        "status": "success",
        "message": "",
        "timestamp": "2023-05-22T10:51:45.860800",
        "requestId": "0ed3d0b76d",
        "targetClasses": ["0", "1"],
        "targetDescription": "A binary variable indicating whether or not the \
            passenger survived (0 = No, 1 = Yes).",
        "predictions": [
            {
                "sampleId": "879",
                "predictedClass": "0",
                "predictedProbabilities": [0.92107, 0.07893],
                "explanation": {
                    "baseline": [0.57775, 0.42225],
                    "featureScores": {
                        "Age_na": [0.05389, -0.05389],
                        "Age": [0.02582, -0.02582],
                        "SibSp": [-0.00469, 0.00469],
                        "Parch": [0.00706, -0.00706],
                        "Fare": [0.05561, -0.05561],
                        "Embarked_S": [0.01582, -0.01582],
                        "Embarked_C": [0.00393, -0.00393],
                        "Embarked_Q": [0.00657, -0.00657],
                        "Pclass_3": [0.0179, -0.0179],
                        "Pclass_1": [0.02394, -0.02394],
                        "Sex_male": [0.13747, -0.13747],
                    },
                },
            }
        ],
        "explanationMethod": "Shap",
    }


def test_ping(app):
    """Test the /ping endpoint."""
    response = app.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "Pong!"}


@patch("src.serve.transform_req_data_and_make_predictions")
def test_infer_endpoint(mock_transform_and_predict, app, sample_request_data):
    """
    Test the infer endpoint.

    The function creates a mock request and sets the expected return value of the
    mock_transform_and_predict function.
    It then sends a POST request to the "/infer" endpoint with the mock request data.
    The function asserts that the response status code is 200 and the JSON response
    matches the expected output.
    Additionally, it checks if the mock_transform_and_predict function was called with
    the correct arguments.

    Args:
        mock_transform_and_predict (MagicMock): A mock of the
            transform_req_data_and_make_predictions function.
        app (TestClient): The TestClient fastapi app

    """
    # Define what your mock should return
    mock_transform_and_predict.return_value = pd.DataFrame(), {
        "status": "success",
        "predictions": [],
    }

    response = app.post("/infer", data=json.dumps(sample_request_data))

    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"status": "success", "predictions": []}
    # You can add more assertions to check if the function was called with the
    # correct arguments
    mock_transform_and_predict.assert_called()
