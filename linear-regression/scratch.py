import sklearn.datasets
import pandas as pd


def mean_squared_error(predicted: list[float], actual: list[float]) -> float:
    sum = 0
    for y_hat, y in zip(predicted, actual):
        y_diff = y_hat - y
        sum += y_diff * y_diff
    return sum / len(predicted)


W = float(input("Enter weight (W): "))
B = float(input("Enter bias (B): "))
DATASET: pd.DataFrame = sklearn.datasets.fetch_california_housing(as_frame=True)


dataset_analysis = DATASET.frame.describe()
print("Dataset Analysis:")
print(dataset_analysis)


def optimise(iterations: int) -> None:
    predicted_values: list[float] = []
    actual_values: list[float] = []

    for _, row in DATASET.frame.iterrows():
        x = row["MedInc"]
        y = row["MedHouseVal"]

        y_hat = W * x + B

        predicted_values.append(y_hat)
        actual_values.append(y)

    mse = mean_squared_error(predicted_values, actual_values)
    print(f"Mean Squared Error: {mse}")

    for _ in range(iterations - 1):
        # TODO: Implement weight and bias updates using gradient descent
        # Look up https://www.youtube.com/watch?v=PaFPbb66DxQ for linear regression
        # Look up https://www.youtube.com/watch?v=sDv4f4s2SB8 for gradient descent
        pass
