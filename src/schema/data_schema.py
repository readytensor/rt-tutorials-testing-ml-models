from typing import List, Dict, Tuple
import joblib

from utils import read_json_as_dict
from data_models.schema_validator import validate_schema_dict


class BinaryClassificationSchema:
    """
    A class for loading and providing access to a binary classification schema.

    This class allows users to work with a generic schema for binary classification
    problems, enabling them to create algorithm implementations that are not hardcoded
    to specific feature names. The class provides methods to retrieve information about
    the schema, such as the ID field, target field, allowed values for the target
    field, and details of the features (categorical and numeric). This makes it easier
    to preprocess and manipulate the input data according to the schema, regardless of
    the specific dataset used.
    """

    def __init__(self, schema_dict: dict) -> None:
        """
        Initializes a new instance of the `BinaryClassificationSchema` class
        and using the schema dictionary.

        Args:
            schema_dict (dict): The python dictionary of schema.
        """
        self.schema = schema_dict
        self._numeric_features, self._categorical_features = self._get_features()

    @property
    def model_category(self) -> str:
        """
        Gets the model category.

        Returns:
            str: The category of the machine learning model
                (e.g., binary_classification, multi-class_classification,
                regression, object_detection, etc.).
        """
        return self.schema["modelCategory"]

    @property
    def title(self) -> str:
        """
        Gets the title of the dataset or problem.

        Returns:
            str: The title of the dataset or the problem.
        """
        return self.schema["title"]

    @property
    def description(self) -> str:
        """
        Gets the description of the dataset or problem.

        Returns:
            str: A brief description of the dataset or the problem.
        """
        return self.schema["description"]

    @property
    def schema_version(self) -> float:
        """
        Gets the version number of the schema.

        Returns:
            float: The version number of the schema.
        """
        return self.schema["schemaVersion"]

    @property
    def input_data_format(self) -> str:
        """
        Gets the format of the input data.

        Returns:
            str: The format of the input data (e.g., CSV, JSON, etc.).
        """
        return self.schema["inputDataFormat"]

    def _get_features(self) -> Tuple[List[str], List[str]]:
        """
        Returns the feature names of numeric and categorical data types.

        Returns:
            Tuple[List[str], List[str]]: The list of numeric feature names, and the
                list of categorical feature names.
        """
        fields = self.schema["features"]
        numeric_features = [f["name"] for f in fields if f["dataType"] == "NUMERIC"]
        categorical_features = [
            f["name"] for f in fields if f["dataType"] == "CATEGORICAL"
        ]
        return numeric_features, categorical_features

    @property
    def id(self) -> str:
        """
        Gets the name of the ID field.

        Returns:
            str: The name of the ID field.
        """
        return self.schema["id"]["name"]

    @property
    def id_description(self) -> str:
        """
        Gets the description for the ID field.

        Returns:
            str: The description for the ID field.
        """
        return self.schema["id"].get(
            "description", "No description for target available."
        )

    @property
    def target(self) -> str:
        """
        Gets the name of the target field.

        Returns:
            str: The name of the target field.
        """
        return self.schema["target"]["name"]

    @property
    def target_classes(self) -> List[str]:
        """
        Gets the classes for the target field.

        Returns:
            List[str]: The list of allowed classes for the target field.
        """
        return [str(c) for c in self.schema["target"]["classes"]]

    @property
    def positive_class(self) -> str:
        """
        Gets the positive class for the target.

        Returns:
            str: The positive class for the target.
        """
        return str(self.schema["target"]["classes"][1])

    @property
    def target_description(self) -> str:
        """
        Gets the description for the target field.

        Returns:
            str: The description for the target field.
        """
        return self.schema["target"].get(
            "description", "No description for target available."
        )

    @property
    def numeric_features(self) -> List[str]:
        """
        Gets the names of the numeric features.

        Returns:
            List[str]: The list of numeric feature names.
        """
        return self._numeric_features

    @property
    def categorical_features(self) -> List[str]:
        """
        Gets the names of the categorical features.

        Returns:
            List[str]: The list of categorical feature names.
        """
        return self._categorical_features

    @property
    def allowed_categorical_values(self) -> Dict[str, List[str]]:
        """
        Gets the allowed values for the categorical features.

        Returns:
            Dict[str, List[str]]: A dictionary of categorical feature names and their
                corresponding allowed values.
        """
        features = self.schema["features"]
        allowed_values = {}
        for feature in features:
            if feature["dataType"] == "CATEGORICAL":
                allowed_values[feature["name"]] = feature["categories"]
        return allowed_values

    def get_allowed_values_for_categorical_feature(
        self, feature_name: str
    ) -> List[str]:
        """
        Gets the allowed values for a single categorical feature.

        Args:
            feature_name (str): The name of the categorical feature.

        Returns:
            List[str]: The list of allowed values for the specified
                categorical feature.
        """
        features = self.schema["features"]
        for feature in features:
            if feature["dataType"] == "CATEGORICAL" and feature["name"] == feature_name:
                return feature["categories"]
        raise ValueError(
            f"Categorical feature '{feature_name}' not found in the schema."
        )

    def get_description_for_feature(self, feature_name: str) -> str:
        """
        Gets the description for a single feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            str: The description for the specified feature.
        """
        fields = self.schema["features"]
        for field in fields:
            if field["name"] == feature_name:
                return field.get("description", "No description for feature available.")
        raise ValueError(f"Feature '{feature_name}' not found in the schema.")

    def get_example_value_for_feature(self, feature_name: str) -> List[str]:
        """
        Gets the example value for a single feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            List[str]: The example values for the specified feature.
        """

        fields = self.schema["features"]
        for field in fields:
            if field["name"] == feature_name:
                if field["dataType"] == "NUMERIC":
                    return field.get("example", 0.0)
                elif field["dataType"] == "CATEGORICAL":
                    return field["categories"][0]
                else:
                    raise ValueError(
                        "Invalid data type for Feature"
                        f"'{feature_name}' found in the schema."
                    )
        raise ValueError(f"Feature '{feature_name}' not found in the schema.")

    def is_feature_nullable(self, feature_name: str) -> bool:
        """
        Check if a feature is nullable.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            bool: True if the feature is nullable, False otherwise.
        """
        fields = self.schema["features"]
        for field in fields:
            if field["name"] == feature_name:
                return field.get("nullable", False)
        raise ValueError(f"Feature '{feature_name}' not found in the schema.")

    @property
    def features(self) -> List[str]:
        """
        Gets the names of all the features.

        Returns:
            List[str]: The list of all feature names (numeric and categorical).
        """
        return self.numeric_features + self.categorical_features

    @property
    def all_fields(self) -> List[str]:
        """
        Gets the names of all the fields.

        Returns:
            List[str]: The list of all field names (ID field, target field, and
                all features).
        """
        return [self.id, self.target] + self.features


def load_json_data_schema(schema_dir_path: str) -> BinaryClassificationSchema:
    """
    Load the JSON file schema into a dictionary, validate the schema dict for
    its correctness, and use the validated schema to instantiate the schema provider.

    Args:
    - schema_dir_path (str): Path from where to read the schema json file.

    Returns:
        BinaryClassificationSchema: An instance of the BinaryClassificationSchema.
    """
    schema_dict = read_json_as_dict(input_path=schema_dir_path)
    validated_schema_dict = validate_schema_dict(schema_dict=schema_dict)
    data_schema = BinaryClassificationSchema(validated_schema_dict)
    return data_schema


def save_schema(schema: BinaryClassificationSchema, output_path: str) -> None:
    """
    Save the schema to a JSON file.

    Args:
        schema (BinaryClassificationSchema): The schema to be saved.
        output_path (str): The path to save the schema to.
    """
    joblib.dump(schema, output_path)


def load_saved_schema(input_path: str) -> BinaryClassificationSchema:
    """
    Load the saved schema from a JSON file.

    Args:
        input_path (str): The path to load the schema from.

    Returns:
        BinaryClassificationSchema: An instance of the BinaryClassificationSchema.
    """
    return joblib.load(input_path)
