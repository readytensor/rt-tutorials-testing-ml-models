import os

import pytest
from py._path.local import LocalPath

from src.train import run_training


@pytest.mark.slow
@pytest.mark.parametrize("run_tuning", [False, True])
def test_run_training(
    run_tuning: bool,
    tmpdir: LocalPath,
    input_schema_dir: str,
    model_config_file_path: str,
    train_dir: str,
    pipeline_config_file_path: str,
    default_hyperparameters_file_path: str,
    hpt_specs_file_path: str,
    explainer_config_file_path: str,
) -> None:
    """Test the run_training function to make sure it produces the required artifacts.

    This test function checks whether the run_training function runs end-to-end
    without errors and produces the expected artifacts. It does this by running
    the training process with and without hyperparameter tuning. After each run,
    it verifies that the expected artifacts have been saved to disk at the correct
    paths.

    Args:
        run_tuning (bool): Boolean indicating whether to run hyperparameter
            tuning or not.
        tmpdir (LocalPath): Temporary directory path provided by the pytest fixture.
        input_schema_dir (str): Path to the input schema directory.
        model_config_file_path (str): Path to the model configuration file.
        train_dir (str): Path to the training directory.
        pipeline_config_file_path (str): Path to the pipeline configuration file.
        default_hyperparameters_file_path (str): Path to the default
            hyperparameters file.
        hpt_specs_file_path (str): Path to the hyperparameter tuning
            specifications file.
        explainer_config_file_path (str): Path to the explainer configuration
            file.
    """
    # Create temporary paths
    saved_schema_path = str(tmpdir.join("saved_schema.json"))
    pipeline_file_path = str(tmpdir.join("pipeline.joblib"))
    target_encoder_file_path = str(tmpdir.join("target_encoder.joblib"))
    predictor_file_path = str(tmpdir.join("predictor.joblib"))
    hpt_results_file_path = str(tmpdir.join("hpt_results.csv"))
    explainer_file_path = str(tmpdir.join("explainer.joblib"))

    # Run the training process without tuning
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_path=saved_schema_path,
        model_config_file_path=model_config_file_path,
        train_dir=train_dir,
        pipeline_config_file_path=pipeline_config_file_path,
        pipeline_file_path=pipeline_file_path,
        target_encoder_file_path=target_encoder_file_path,
        predictor_file_path=predictor_file_path,
        default_hyperparameters_file_path=default_hyperparameters_file_path,
        run_tuning=run_tuning,
        hpt_specs_file_path=hpt_specs_file_path if run_tuning else None,
        hpt_results_file_path=hpt_results_file_path if run_tuning else None,
        explainer_config_file_path=explainer_config_file_path,
        explainer_file_path=explainer_file_path,
    )

    # Assert that the model artifacts are saved in the correct paths
    assert os.path.isfile(saved_schema_path)
    assert os.path.isfile(pipeline_file_path)
    assert os.path.isfile(target_encoder_file_path)
    assert os.path.isfile(predictor_file_path)
    assert os.path.isfile(explainer_file_path)
    if run_tuning:
        assert os.path.isfile(hpt_results_file_path)
