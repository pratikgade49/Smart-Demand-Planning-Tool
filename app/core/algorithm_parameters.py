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
                )
            ]
        },
        2: {  # Linear Regression
            "algorithm_name": "Linear Regression",
            "description": "Simple linear regression forecasting",
            "parameters": []
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
                    name="n_estimators_list",
                    type="list",
                    description="Number of estimators to try",
                    required=True,
                    default_value=[50, 100],
                    list_item_type="int",
                    min_value=10,
                    max_value=1000
                ),
                ParameterDefinition(
                    name="learning_rate_list",
                    type="list",
                    description="Learning rates to try",
                    required=True,
                    default_value=[0.05, 0.1, 0.2],
                    list_item_type="float",
                    min_value=0.01,
                    max_value=1.0
                ),
                ParameterDefinition(
                    name="max_depth_list",
                    type="list",
                    description="Maximum depths to try",
                    required=True,
                    default_value=[3, 4, 5],
                    list_item_type="int",
                    min_value=1,
                    max_value=10
                )
            ]
        },
        10: {  # SVR
            "algorithm_name": "SVR",
            "description": "Support Vector Regression",
            "parameters": [
                ParameterDefinition(
                    name="C_list",
                    type="list",
                    description="Regularization parameters to try",
                    required=True,
                    default_value=[1, 10, 100],
                    list_item_type="float",
                    min_value=0.1,
                    max_value=1000.0
                ),
                ParameterDefinition(
                    name="epsilon_list",
                    type="list",
                    description="Epsilon values to try",
                    required=True,
                    default_value=[0.1, 0.2],
                    list_item_type="float",
                    min_value=0.01,
                    max_value=1.0
                )
            ]
        },
        11: {  # KNN
            "algorithm_name": "KNN",
            "description": "K-Nearest Neighbors regression",
            "parameters": [
                ParameterDefinition(
                    name="n_neighbors_list",
                    type="list",
                    description="Number of neighbors to try",
                    required=True,
                    default_value=[7, 10],
                    list_item_type="int",
                    min_value=1,
                    max_value=50
                )
            ]
        },
        12: {  # Gaussian Process
            "algorithm_name": "Gaussian Process",
            "description": "Gaussian Process regression",
            "parameters": []
        },
        13: {  # Neural Network
            "algorithm_name": "Neural Network",
            "description": "Multi-layer perceptron neural network",
            "parameters": [
                ParameterDefinition(
                    name="hidden_layer_sizes_list",
                    type="list",
                    description="Hidden layer configurations to try",
                    required=True,
                    default_value=[[10], [20, 10]],
                    list_item_type="list",
                    min_value=1,
                    max_value=100
                ),
                ParameterDefinition(
                    name="alpha_list",
                    type="list",
                    description="Regularization parameters to try",
                    required=True,
                    default_value=[0.001, 0.01],
                    list_item_type="float",
                    min_value=0.0001,
                    max_value=1.0
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
        result = ParameterValidationResult(is_valid=True, errors=[], warnings=[])

        # Get parameter schema
        schema = AlgorithmParametersService.get_algorithm_parameters(algorithm_id)
        if not schema:
            result.is_valid = False
            result.errors.append(f"Unknown algorithm ID: {algorithm_id}")
            return result

        if not custom_parameters:
            # Check if any required parameters are missing
            for param in schema.parameters:
                if param.required:
                    result.is_valid = False
                    result.errors.append(f"Required parameter '{param.name}' is missing")
            return result

        # Validate each parameter
        for param in schema.parameters:
            param_name = param.name
            if param_name not in custom_parameters:
                if param.required:
                    result.is_valid = False
                    result.errors.append(f"Required parameter '{param_name}' is missing")
                continue

            param_value = custom_parameters[param_name]

            # Type validation
            if not AlgorithmParametersService._validate_parameter_type(param_value, param.type):
                result.is_valid = False
                result.errors.append(f"Parameter '{param_name}' must be of type {param.type}")
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

        return isinstance(value, expected_python_type)
