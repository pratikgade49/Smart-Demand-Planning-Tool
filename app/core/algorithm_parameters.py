"""
Algorithm Parameters Service.
Provides parameter schemas and validation for forecasting algorithms.
"""

from typing import Dict, List, Any, Optional
import logging
from app.schemas.forecasting import (
    ParameterDefinition,
    AlgorithmParameterSchema,
    ParameterValidationResult
)

logger = logging.getLogger(__name__)


class AlgorithmParametersService:
    """Service for managing algorithm parameter schemas and validation."""

    # Algorithm parameter definitions
    ALGORITHM_PARAMETERS = {
        1: {  # ARIMA
            "algorithm_name": "ARIMA",
            "description": "AutoRegressive Integrated Moving Average - Statistical time series forecasting",
            "parameters": [
                ParameterDefinition(
                    name="order",
                    type="list",
                    description="ARIMA order (p, d, q) - autoregressive, differencing, moving average components",
                    required=True,
                    default_value=[1, 1, 1],
                    list_item_type="int",
                    min_value=0,
                    max_value=10
                ),
                ParameterDefinition(
                    name="seasonal_order",
                    type="list",
                    description="ARIMA seasonal order (P, D, Q, s)",
                    required=False,
                    default_value=[0, 0, 0, 0],
                    list_item_type="int",
                    min_value=0,
                    max_value=12
                )
            ]
        },
        2: {  # Linear Regression
            "algorithm_name": "Linear Regression",
            "description": "Simple linear regression forecasting",
            "parameters": [
                ParameterDefinition(
                    name="fit_intercept",
                    type="bool",
                    description="Whether to calculate the intercept for this model",
                    required=False,
                    default_value=True
                )
            ]
        },
        3: {  # Polynomial Regression
            "algorithm_name": "Polynomial Regression",
            "description": "Polynomial regression with configurable degree",
            "parameters": [
                ParameterDefinition(
                    name="degree",
                    type="int",
                    description="Polynomial degree for regression",
                    required=True,
                    default_value=2,
                    min_value=1,
                    max_value=5
                )
            ]
        },
        4: {  # Exponential Smoothing
            "algorithm_name": "Exponential Smoothing",
            "description": "Simple exponential smoothing",
            "parameters": [
                ParameterDefinition(
                    name="alpha",
                    type="float",
                    description="Smoothing factor (0-1)",
                    required=True,
                    default_value=0.3,
                    min_value=0.0,
                    max_value=1.0
                )
            ]
        },
        5: {  # Enhanced Exponential Smoothing
            "algorithm_name": "Enhanced Exponential Smoothing",
            "description": "Multiple alpha values for enhanced smoothing",
            "parameters": [
                ParameterDefinition(
                    name="alphas",
                    type="list",
                    description="List of smoothing factors to try",
                    required=True,
                    default_value=[0.1, 0.3, 0.5],
                    list_item_type="float",
                    min_value=0.0,
                    max_value=1.0
                )
            ]
        },
        6: {  # Holt Winters
            "algorithm_name": "Holt Winters",
            "description": "Triple exponential smoothing with seasonality",
            "parameters": [
                ParameterDefinition(
                    name="season_length",
                    type="int",
                    description="Seasonal period length",
                    required=True,
                    default_value=12,
                    min_value=2,
                    max_value=365
                )
            ]
        },
        7: {  # Prophet
            "algorithm_name": "Prophet",
            "description": "Facebook Prophet forecasting model",
            "parameters": [
                ParameterDefinition(
                    name="window",
                    type="int",
                    description="Window size for trend detection",
                    required=True,
                    default_value=3,
                    min_value=1,
                    max_value=30
                ),
                ParameterDefinition(
                    name="changepoint_prior_scale",
                    type="float",
                    description="Flexibility of the automatic changepoint selection",
                    required=False,
                    default_value=0.05,
                    min_value=0.001,
                    max_value=0.5
                )
            ]
        },
        8: {  # LSTM Neural Network
            "algorithm_name": "LSTM Neural Network",
            "description": "Long Short-Term Memory neural network",
            "parameters": [
                ParameterDefinition(
                    name="window",
                    type="int",
                    description="Lookback window size",
                    required=True,
                    default_value=3,
                    min_value=1,
                    max_value=50
                )
            ]
        },
        9: {  # XGBoost
            "algorithm_name": "XGBoost",
            "description": "Extreme Gradient Boosting",
            "parameters": [
                ParameterDefinition(
                    name="n_estimators",
                    type="int",
                    description="Number of boosting rounds",
                    required=False,
                    default_value=100,
                    min_value=10,
                    max_value=1000
                ),
                ParameterDefinition(
                    name="max_depth",
                    type="int",
                    description="Maximum tree depth",
                    required=False,
                    default_value=6,
                    min_value=1,
                    max_value=20
                ),
                ParameterDefinition(
                    name="learning_rate",
                    type="float",
                    description="Boosting learning rate",
                    required=False,
                    default_value=0.1,
                    min_value=0.01,
                    max_value=1.0
                )
            ]
        },
        10: {  # SVR
            "algorithm_name": "SVR",
            "description": "Support Vector Regression",
            "parameters": [
                ParameterDefinition(
                    name="C",
                    type="float",
                    description="Regularization parameter",
                    required=False,
                    default_value=1.0,
                    min_value=0.1,
                    max_value=1000.0
                ),
                ParameterDefinition(
                    name="epsilon",
                    type="float",
                    description="Epsilon-tube parameter",
                    required=False,
                    default_value=0.1,
                    min_value=0.01,
                    max_value=1.0
                ),
                ParameterDefinition(
                    name="kernel",
                    type="string",
                    description="Kernel type",
                    required=False,
                    default_value="rbf",
                    allowed_values=["linear", "poly", "rbf", "sigmoid"]
                )
            ]
        },
        11: {  # KNN
            "algorithm_name": "KNN",
            "description": "K-Nearest Neighbors regression",
            "parameters": [
                ParameterDefinition(
                    name="n_neighbors",
                    type="int",
                    description="Number of neighbors to use",
                    required=False,
                    default_value=5,
                    min_value=1,
                    max_value=50
                )
            ]
        },
        12: {  # Gaussian Process
            "algorithm_name": "Gaussian Process",
            "description": "Gaussian Process regression",
            "parameters": [
                ParameterDefinition(
                    name="alpha",
                    type="float",
                    description="Value added to the diagonal of the kernel matrix during fitting",
                    required=False,
                    default_value=1e-10,
                    min_value=1e-15,
                    max_value=1.0
                )
            ]
        },
        13: {  # Neural Network
            "algorithm_name": "Neural Network",
            "description": "Multi-layer perceptron neural network",
            "parameters": [
                ParameterDefinition(
                    name="hidden_layers",
                    type="list",
                    description="Hidden layer sizes",
                    required=False,
                    default_value=[64, 32],
                    list_item_type="int",
                    min_value=1,
                    max_value=100
                ),
                ParameterDefinition(
                    name="epochs",
                    type="int",
                    description="Number of training epochs",
                    required=False,
                    default_value=100,
                    min_value=1,
                    max_value=1000
                ),
                ParameterDefinition(
                    name="batch_size",
                    type="int",
                    description="Batch size for training",
                    required=False,
                    default_value=32,
                    min_value=1,
                    max_value=256
                )
            ]
        },
        14: {  # Moving Average
            "algorithm_name": "Moving Average",
            "description": "Simple moving average forecasting",
            "parameters": [
                ParameterDefinition(
                    name="window",
                    type="int",
                    description="Window size for moving average calculation",
                    required=True,
                    default_value=3,
                    min_value=1,
                    max_value=50
                )
            ]
        },
        15: {  # SARIMA
            "algorithm_name": "SARIMA",
            "description": "Seasonal AutoRegressive Integrated Moving Average - Statistical time series forecasting with seasonality",
            "parameters": [
                ParameterDefinition(
                    name="order",
                    type="list",
                    description="SARIMA order (p, d, q) - autoregressive, differencing, moving average components",
                    required=True,
                    default_value=[1, 1, 1],
                    list_item_type="int",
                    min_value=0,
                    max_value=10
                ),
                ParameterDefinition(
                    name="seasonal_order",
                    type="list",
                    description="SARIMA seasonal order (P, D, Q, s)",
                    required=True,
                    default_value=[1, 1, 1, 12],
                    list_item_type="int",
                    min_value=0,
                    max_value=12
                )
            ]
        },
        16: {  # Random Forest
            "algorithm_name": "Random Forest",
            "description": "Random Forest regression with ensemble learning",
            "parameters": [
                ParameterDefinition(
                    name="n_estimators",
                    type="int",
                    description="Number of trees in the forest",
                    required=True,
                    default_value=100,
                    min_value=10,
                    max_value=500
                ),
                ParameterDefinition(
                    name="max_depth",
                    type="int",
                    description="Maximum depth of the trees (None for unlimited)",
                    required=False,
                    default_value=None,
                    min_value=1,
                    max_value=50
                ),
                ParameterDefinition(
                    name="min_samples_split",
                    type="int",
                    description="Minimum number of samples required to split an internal node",
                    required=True,
                    default_value=2,
                    min_value=2,
                    max_value=20
                ),
                ParameterDefinition(
                    name="min_samples_leaf",
                    type="int",
                    description="Minimum number of samples required to be at a leaf node",
                    required=True,
                    default_value=1,
                    min_value=1,
                    max_value=10
                )
            ]
        },
        999: {  # Best Fit
            "algorithm_name": "Best Fit",
            "description": "Auto-selection of best performing algorithm",
            "parameters": []
        }
    }

    @staticmethod
    def get_algorithm_parameters(algorithm_id: int) -> Optional[AlgorithmParameterSchema]:
        """
        Get parameter schema for a specific algorithm.

        Args:
            algorithm_id: The algorithm ID

        Returns:
            AlgorithmParameterSchema if found, None otherwise
        """
        if algorithm_id not in AlgorithmParametersService.ALGORITHM_PARAMETERS:
            return None

        params_data = AlgorithmParametersService.ALGORITHM_PARAMETERS[algorithm_id]
        return AlgorithmParameterSchema(
            algorithm_id=algorithm_id,
            algorithm_name=params_data["algorithm_name"],
            parameters=params_data["parameters"],
            description=params_data.get("description")
        )

    @staticmethod
    def get_all_algorithm_parameters() -> List[AlgorithmParameterSchema]:
        """
        Get parameter schemas for all algorithms.

        Returns:
            List of AlgorithmParameterSchema objects
        """
        schemas = []
        for algorithm_id in AlgorithmParametersService.ALGORITHM_PARAMETERS:
            schema = AlgorithmParametersService.get_algorithm_parameters(algorithm_id)
            if schema:
                schemas.append(schema)
        return schemas

    @staticmethod
    def validate_parameters(
        algorithm_id: int,
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> ParameterValidationResult:
        """
        Validate custom parameters against algorithm schema.

        Args:
            algorithm_id: The algorithm ID
            custom_parameters: Custom parameters to validate

        Returns:
            ParameterValidationResult with validation status and messages
        """
        result = ParameterValidationResult(
            is_valid=True, 
            errors=[], 
            warnings=[],
            validated_parameters={}
        )

        # Get parameter schema
        schema = AlgorithmParametersService.get_algorithm_parameters(algorithm_id)
        if not schema:
            result.is_valid = False
            result.errors.append(f"Unknown algorithm ID: {algorithm_id}")
            return result

        input_params = custom_parameters or {}
        validated_params = {}

        # Validate each parameter defined in schema
        for param in schema.parameters:
            param_name = param.name
            
            if param_name not in input_params:
                if param.required:
                    if param.default_value is not None:
                        # Use default value if required but missing
                        validated_params[param_name] = param.default_value
                        result.warnings.append(f"Required parameter '{param_name}' missing, using default value")
                    else:
                        result.is_valid = False
                        result.errors.append(f"Required parameter '{param_name}' is missing")
                else:
                    # Use default value for optional parameters if missing
                    validated_params[param_name] = param.default_value
                continue

            param_value = input_params[param_name]

            # Handle None values for optional parameters
            if param_value is None:
                if param.required:
                    result.is_valid = False
                    result.errors.append(f"Required parameter '{param_name}' cannot be None")
                else:
                    validated_params[param_name] = param_value
                continue

            # Type validation
            if not AlgorithmParametersService._validate_parameter_type(param_value, param.type):
                result.is_valid = False
                result.errors.append(f"Parameter '{param_name}' must be of type {param.type}")
                continue

            # Convert string values to expected types
            if isinstance(param_value, str):
                try:
                    if param.type == 'int':
                        param_value = int(param_value)
                    elif param.type == 'float':
                        param_value = float(param_value)
                    elif param.type == 'bool':
                        param_value = param_value.lower() in ['true', '1']
                except ValueError:
                    result.is_valid = False
                    result.errors.append(f"Parameter '{param_name}' cannot be converted to {param.type}")
                    continue

            # Range validation for numeric types
            if param.type in ['int', 'float']:
                if param.min_value is not None and param_value < param.min_value:
                    result.is_valid = False
                    result.errors.append(f"Parameter '{param_name}' must be >= {param.min_value}")
                if param.max_value is not None and param_value > param.max_value:
                    result.is_valid = False
                    result.errors.append(f"Parameter '{param_name}' must be <= {param.max_value}")

            # List validation
            elif param.type == 'list':
                if not isinstance(param_value, list):
                    result.is_valid = False
                    result.errors.append(f"Parameter '{param_name}' must be a list")
                    continue

                if param.list_item_type:
                    for i, item in enumerate(param_value):
                        if not AlgorithmParametersService._validate_parameter_type(item, param.list_item_type):
                            result.is_valid = False
                            result.errors.append(f"List item {i} in '{param_name}' must be of type {param.list_item_type}")
                            break

                # Range validation for list items
                if param.list_item_type in ['int', 'float']:
                    for i, item in enumerate(param_value):
                        if param.min_value is not None and item < param.min_value:
                            result.is_valid = False
                            result.errors.append(f"List item {i} in '{param_name}' must be >= {param.min_value}")
                        if param.max_value is not None and item > param.max_value:
                            result.is_valid = False
                            result.errors.append(f"List item {i} in '{param_name}' must be <= {param.max_value}")
                elif param.list_item_type == 'list':
                    for i, sublist in enumerate(param_value):
                        if isinstance(sublist, list):
                            for j, item in enumerate(sublist):
                                if isinstance(item, (int, float)):
                                    if param.min_value is not None and item < param.min_value:
                                        result.is_valid = False
                                        result.errors.append(f"Item {j} in list {i} of '{param_name}' must be >= {param.min_value}")
                                    if param.max_value is not None and item > param.max_value:
                                        result.is_valid = False
                                        result.errors.append(f"Item {j} in list {i} of '{param_name}' must be <= {param.max_value}")

            # Allowed values validation
            if param.allowed_values is not None:
                if param.type == 'list':
                    for item in param_value:
                        if item not in param.allowed_values:
                            result.is_valid = False
                            result.errors.append(f"Value '{item}' in '{param_name}' is not allowed")
                            break
                else:
                    if param_value not in param.allowed_values:
                        result.is_valid = False
                        result.errors.append(f"Value '{param_value}' for '{param_name}' is not allowed")

            validated_params[param_name] = param_value

        result.validated_parameters = validated_params
        return result

    @staticmethod
    def _validate_parameter_type(value: Any, expected_type: str) -> bool:
        """
        Validate parameter type.

        Args:
            value: The value to validate
            expected_type: Expected type string

        Returns:
            True if valid, False otherwise
        """
        type_map = {
            'int': int,
            'float': float,
            'string': str,
            'bool': bool,
            'list': list
        }

        expected_python_type = type_map.get(expected_type)
        if not expected_python_type:
            return False

        # Allow int for float types
        if expected_type == 'float' and isinstance(value, int):
            return True

        # Allow string representations of numbers
        if isinstance(value, str):
            try:
                if expected_type == 'int':
                    int(value)
                    return True
                elif expected_type == 'float':
                    float(value)
                    return True
                elif expected_type == 'bool':
                    value.lower() in ['true', 'false', '1', '0']
                    return True
            except ValueError:
                return False

        return isinstance(value, expected_python_type)
