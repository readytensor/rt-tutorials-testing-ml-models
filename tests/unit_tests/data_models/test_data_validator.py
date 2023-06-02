from typing import Any

import pandas as pd
import pytest

from src.data_models.data_validator import validate_data


def test_validate_data(
    schema_provider: Any,
    sample_train_data: pd.DataFrame,
    sample_test_data: pd.DataFrame,
):
    """
    Tests the `validate_data` function.

    It checks the function for several scenarios:
    1. When the input DataFrame (either train or test data) is correctly formatted
        according to the schema, no error should be raised, and the returned DataFrame
        should be identical to the input DataFrame.
    2. When a required column (according to the schema) is missing from the input
        DataFrame, a ValueError should be raised.

    Args:
        schema_provider (BinaryClassificationSchema): The schema provider instance which
            encapsulates the data schema.
        sample_train_data (pd.DataFrame): A sample training DataFrame formatted
            correctly according to the schema.
        sample_test_data (pd.DataFrame): A sample testing DataFrame formatted correctly
            according to the schema.

    Raises:
        pytest.fail: If an unexpected error is raised during validation of correct data.
    """
    # Test with correct data - train data, has target
    try:
        result_train_data = validate_data(sample_train_data, schema_provider, True)
        # check if train DataFrame is unchanged
        pd.testing.assert_frame_equal(result_train_data, sample_train_data)
    except AssertionError as exc:
        pytest.fail(
            f"Returned DataFrame is not identical to the input DataFrame: {exc}"
        )

    # Test with correct data - test data, doesnt have target
    try:
        result_test_data = validate_data(sample_test_data, schema_provider, False)
        # check if test DataFrame is unchanged
        pd.testing.assert_frame_equal(result_test_data, sample_test_data)
    except AssertionError as exc:
        pytest.fail(
            f"Returned DataFrame is not identical to the input DataFrame: {exc}"
        )

    # Test with incorrect data (missing feature column in train data)
    missing_feature_data = sample_train_data.drop(columns=["numeric_feature_1"])
    with pytest.raises(ValueError):
        validate_data(missing_feature_data, schema_provider, True)

    # Test with incorrect data (missing feature column in test data)
    missing_feature_data = sample_test_data.drop(columns=["numeric_feature_1"])
    with pytest.raises(ValueError):
        validate_data(missing_feature_data, schema_provider, False)

    # Test with incorrect data (missing id column in train data)
    missing_id_data = sample_train_data.drop(columns=["id"])
    with pytest.raises(ValueError):
        validate_data(missing_id_data, schema_provider, True)

    # Test with incorrect data (missing id column in test data)
    missing_id_data = sample_test_data.drop(columns=["id"])
    with pytest.raises(ValueError):
        validate_data(missing_id_data, schema_provider, True)

    # Test with incorrect data (missing target column and is_train=True)
    missing_target_data = sample_train_data.drop(columns=["target_field"])
    with pytest.raises(ValueError):
        validate_data(missing_target_data, schema_provider, True)
