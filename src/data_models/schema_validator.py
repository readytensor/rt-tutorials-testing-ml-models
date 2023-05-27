from typing import List, Union, Optional
from pydantic import BaseModel, validator, ValidationError
from enum import Enum


class ID(BaseModel):
    """
    A model representing the ID field of the dataset.
    """
    name: str
    description: str


class Target(BaseModel):
    """
    A model representing the target field of a binary classification problem.
    """
    name: str
    description: str
    classes: List[str]

    @validator("classes")
    def target_classes_are_two_and_unique_and_not_empty_str(cls, target_classes):
        if len(target_classes) != 2:
            raise ValueError(f"Target classes must be a list with two labels. Given `{target_classes}`")
        if len(set(target_classes)) != 2:
            raise ValueError(f"Target classes must be a list with two unique labels. Given `{target_classes}`")
        if "" in target_classes:
            raise ValueError(f"Target classes must not contain empty strings. Given `{target_classes}`")
        return target_classes


class DataType(str, Enum):
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"


class Feature(BaseModel):
    """
    A model representing the predictor fields in the dataset. Validates the
    presence and type of the 'example' field based on the 'dataType' field
    for NUMERIC dataType and presence and contents of the 'categories' field
    for CATEGORICAL dataType.
    """
    name: str
    description: str
    dataType: DataType
    nullable: bool
    example: Optional[Union[float, str]]
    categories: Optional[List[str]]

    @validator("example", always=True)
    def example_is_present_with_data_type_is_numeric(cls, v, values):
        data_type = values.get("dataType")
        if data_type == "NUMERIC" and v is None:
            raise ValueError(
                f"`example` must be present and a float or an integer when dataType is NUMERIC."
                f"Check field: {values}")
        return v

    @validator("categories", always=True)
    def categories_are_present_with_data_type_is_categorical(cls, v, values):
        data_type = values.get("dataType")
        if data_type == "CATEGORICAL" and v is None:
            raise ValueError(
                f"`categories` must be present when dataType is CATEGORICAL. Check field: {values}", )
        return v

    @validator("categories", always=True)
    def categories_are_non_empty_strings(cls, v, values):
        categories = values.get("categories")
        if categories is not None:
            if len(categories) == 0:
                raise ValueError(f"`categories` must not be empty. Check field: {values}", )
            for category in categories:
                if str(category) == "" or not isinstance(category, str):
                    raise ValueError(f"`categories` must be a list of strings. Check field: {values}", )
        return v



class SchemaModel(BaseModel):
    """
    A schema validator for binary classification problems. Validates the
    problem category, version, and predictor fields of the input schema.
    """
    title: str
    description: str = None
    modelCategory: str
    schemaVersion: float
    inputDataFormat: str = None
    id: ID
    target: Target
    features: List[Feature]

    @validator("modelCategory")
    def valid_problem_category(cls, v):
        if v != "binary_classification":
            raise ValueError(f"modelCategory must be 'binary_classification'. Given {v}")
        return v

    @validator("schemaVersion")
    def valid_version(cls, v):
        if v != 1.0:
            raise ValueError(f"schemaVersion must be set to 1.0. Given {v}")
        return v

    @validator("features")
    def at_least_one_predictor_field(cls, v):
        if len(v) < 1:
            raise ValueError(f"features must have at least one field defined. Given {v}")
        return v


def validate_schema_dict(schema_dict:dict)->None:
    """
        Validate the schema
        Args:
            schema_dict: dict
                data schema as a python dictionary

        Raises:
            ValueError: if the schema is invalid
    """
    try:
        schema_dict = SchemaModel.parse_obj(schema_dict).dict()
        return schema_dict
    except ValidationError as exc:
        raise ValueError(f'Invalid schema: {exc}') from exc


if __name__ == "__main__":
    # test data for the schema
    sample_schema_dict = {
        "title": "test dataset",
        "description": "test dataset description",
        "problemCategory": "binary_classification",
        "version": 1.0,
        "inputDataFormat": "CSV",
        "idField": {
            "name": "id",
            "description": "id field"
        },
        "targetField": {
            "name": "target",
            "description": "test target",
            "allowedValues": ["1", "0"],
            "positiveClass": "1"
        },
        "predictorFields": [
            {
                "name": "test_cat_feature",
                "description": "A categorical variable ",
                "dataType": "CATEGORICAL",
                "allowedValues": ["x", "y", "z"]
            },
            {
                "name": "test_num_feature",
                "description": "A numeric variable",
                "dataType": "NUMERIC",
                "example": 3.14159
            }
        ]
    }

    
    # Create an instance of the Schema model with the schema data
    schema = SchemaModel(**sample_schema_dict)

    # Validate the schema by calling its validate() method
    SchemaModel.validate(sample_schema_dict)

    print("all good")