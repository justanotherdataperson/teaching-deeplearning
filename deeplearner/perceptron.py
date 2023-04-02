"""Perceptron.

Define a perceptron WITHOUT using any third-party packages i.e. just base
python.
"""

# Standard library imports
import random
from typing import List, Optional, Union

# Third party imports
# n/a

# Local application imports
# n/a

NumericType = Union[float, int]
NumericInstance = (float, int)


class Perceptron:
    """Perceptron."""

    def __init__(
        self,
        n_inputs: Optional[int] = None,
        weights: Optional[list] = None,
        learning_rate: float = 0.1,
        epochs: int = 50,
        bias: NumericType = 1.0,
        bias_weight: Optional[NumericType] = None,
        ):
        """Perceptron.

        Args:
            n_inputs (Optional[int], optional): Number of data points (used to
                initialise the appropriate number of weights). Defaults to None.
            weights (Optional[list], optional): Pre-initialised weights.
                Defaults to None.
            learning_rate (float, optional): The learning rate; how quickly the
                algorithm adjusts the weights. Defaults to 0.1.
            epochs (int, optional): Number of epochs; how many times to loop
                through the data whilst training. Defaults to 50.
            bias (NumericType, optional): Bias term (typically 1). Defaults to
                1.0.
            bias_weight (Optional[NumericType], optional): Pre-initialised
                weight for the bias. Defaults to None.
        """
        # Validate inputs...
        assert n_inputs is not None or weights is not None,\
            "either n_inputs or weights must be set"

        if n_inputs is not None:
            assert isinstance(n_inputs, int), "n_inputs must be an integer"

        if weights is not None:
            assert isinstance(weights, list), "weights must be a list"

        assert isinstance(learning_rate, float)
        assert learning_rate > 0, "learning_rate must be greater than zero"
        assert learning_rate < 1, "learning_rate must be less than one"

        assert isinstance(epochs, int), "epochs must be an integer"
        assert epochs > 0, "epochs must be positive"

        assert isinstance(bias, NumericInstance), "bias_weight must be a float"

        if bias_weight is not None:
            assert isinstance(bias_weight, NumericInstance), \
                "bias_weight must be numeric"
            assert bias_weight >= 0, "bias_weight must be >= 0"
            assert bias_weight <= 1, "bias_weight must be >= 0"

        # Extract weights
        self.weights = weights

        self.bias = bias
        self.bias_weight = bias_weight

        # Configurable hyperparameter typically has a small positive value,
        # often in the range between 0.0 and 1.0.
        self.learning_rate = learning_rate

        # The number of times the data is looped through
        self.epochs = epochs

        # Define initial weights
        # How do we pick this? Lots of discussion/research around this
        # For a simple example, we initialise them randomly
        if self.weights is None:

            # Weight for each input
            self.weights = [
                random.random() for _ in range(n_inputs)
                ]

        if self.bias_weight is None:

            self.bias_weight = random.random()

    def train(self, X: List[list], y: list):
        """Train the perceptron.

        Args:
            X (List[list]): Input values
            y (list): Values to predict
        """
        assert isinstance(X, list), "X must be a list"
        assert isinstance(y, list), "y must be a list"

        # Loop through each epoch
        for epoch in range(0, self.epochs):

            # Loop through each training data
            for x, truth in zip(X, y):

                # Run prediction
                prediction = self.feed_forward(X=x)

                new_weights = []

                for _x, w in zip(x, self.weights):

                    weight = self.gradient_descent(
                        x=_x,
                        weight=w,
                        prediction=prediction,
                        truth=truth,
                        )

                    new_weights.append(weight)

                self.weights = new_weights

                self.bias_weight = self.gradient_descent(
                    x=self.bias,
                    weight=self.bias_weight,
                    prediction=prediction,
                    truth=truth,
                )

    def feed_forward(self, X: List[NumericType]) -> int:
        """Forward pass; makes the prediction for a single input.

        Args:
            X (List[NumericType]): Input data

        Returns:
            int: Prediction
        """
        assert isinstance(X, list), "X must be a list"

        vals = X + [self.bias]
        wts = self.weights + [self.bias_weight]

        # Calculate weighted sum of inputs (Neuron calculation)
        weighted_sum = sum([x * w for x, w in zip(vals, wts)])

        # Use activation function to define
        prediction = self.activation_function(input_value=weighted_sum)

        return prediction

    def predict(self, X: List[List[NumericType]]) -> List[int]:
        """Predict the output for a list of inputs.

        Args:
            X (List[List[NumericType]]): Input data

        Returns:
            List[int]: Predictions
        """
        assert isinstance(X, list), "X must be a list"

        return [self.feed_forward(X=x) for x in X]

    def activation_function(self, input_value: NumericType):
        """Convert the weighted sum of inputs to a prediction.

        Extended Summary:
            Determines if the neuron activates or not.
        """
        assert isinstance(input_value, NumericInstance), \
            "input_value must be numeric"

        if (input_value >= 0):
            return 1

        return 0

    def gradient_descent(
        self,
        x: NumericType,
        weight: NumericType,
        prediction: NumericType,
        truth: NumericType,) -> float:
        """Adjust the weights of the perceptron.

        Args:
            x (NumericType): Input value
            weight (NumericType): Weight
            prediction (NumericType): Prediction from the input value
            truth (NumericType): Truth label for the input value

        Returns:
            float: New weight
        """
        assert isinstance(x, NumericInstance), "x must be numeric"
        assert isinstance(weight, NumericInstance), "weight must be numeric"
        assert isinstance(prediction, NumericInstance), \
            "prediction must be numeric"
        assert isinstance(truth, NumericInstance), "truth must be numeric"

        # Get error
        error = truth - prediction

        # Error multiplied by the input mulitplied by the learning rate
        adj_weight = error * x * self.learning_rate

        # Adjust weight
        new_weight = weight + adj_weight

        return new_weight

    def get_boundary(self, X: list) -> list:
        """Get the boundary condition for the binary classification.

        Args:
            X (list): Input x-values

        Returns:
            list: Corresponding y-values for the boundary

        Extended Summary:
            At the boundary...
                (w0 * x) + (w1 * y) + (wb * b) = 0
            So:
                y = - ( (w0 * x) + (wb * b) ) / w1

            This only applies because this is a simple case of a binary
            classification.
        """
        assert isinstance(X, list), "X must be a list"

        w0 = self.weights[0]
        w1 = self.weights[1]
        wb = self.bias_weight

        b = self.bias

        y = [- (w0/w1) * x - b*(wb/w1) for x in X]

        return y
