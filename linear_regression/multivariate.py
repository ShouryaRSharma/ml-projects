from dataclasses import dataclass
from typing import Annotated, Self

import pandas as pd
import sklearn
import typer

app = typer.Typer()


@dataclass
class TrainingData:
    x: pd.DataFrame
    y: pd.Series


@dataclass
class ModelState:
    w: pd.Series
    b: float
    lr: float

    def predict(self: Self, x: pd.DataFrame) -> pd.Series:
        return x.dot(self.w) + self.b

    def step(self: Self, x: pd.DataFrame, y: pd.Series) -> None:
        y_pred = self.predict(x)
        error = y - y_pred

        derivative_b = -2 * error.mean()
        derivative_w = -2 * (x * error.values.reshape(-1, 1)).mean(axis=0)

        self.b -= self.lr * derivative_b
        self.w -= self.lr * derivative_w

    def loss(self: Self, x: pd.DataFrame, y: pd.Series) -> float:
        y_pred = self.predict(x)
        error = y - y_pred
        return (error**2).mean()


def fetch_data(features: list[str]) -> TrainingData:
    dataset = sklearn.datasets.fetch_california_housing(as_frame=True)
    df = dataset.frame
    df = df[df["MedHouseVal"] < 5.0]

    for feature in features:
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = (df[feature] - mean) / std

    print(f"Loaded {len(df)} houses after filtering (removed houses >= $500k)")
    return TrainingData(x=df[features], y=df["MedHouseVal"])


def optimise(
    features: list[str],
    iterations: int,
    learning_rate: float,
    log_interval: int,
    data: TrainingData,
) -> ModelState:
    model = ModelState(
        w=pd.Series(0.0, index=features),
        b=0.0,
        lr=learning_rate,
    )

    for i in range(iterations):
        model.step(data.x, data.y)

        if i % log_interval == 0:
            loss = model.loss(data.x, data.y)
            print(f"Iteration {i:4d} | Loss: {loss:.6f} | B: {model.b:.4f}")
            for feature, weight in model.w.items():
                print(f"  {feature:15s}: {weight:.6f}")

    print("-" * 80)
    print("Training complete!")

    return model


@app.command()
def train(
    features: Annotated[
        list[str],
        typer.Option(
            "--feature",
            "-f",
            help="Feature to use for training (can be specified multiple times)",
        ),
    ] = ["MedInc"],
    iterations: Annotated[
        int, typer.Option("--iterations", "-i", help="Number of training iterations")
    ] = 1000,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learning-rate", "-lr", help="Learning rate for gradient descent"
        ),
    ] = 0.01,
    log_interval: Annotated[
        int, typer.Option("--log-interval", "-l", help="How often to log progress")
    ] = 100,
):
    data = fetch_data(features)
    model = optimise(features, iterations, learning_rate, log_interval, data)
    final_loss = model.loss(data.x, data.y)
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Final B: {model.b:.4f}")
    print("Final Weights:")
    for feature, weight in model.w.items():
        print(f"  {feature:15s}: {weight:.6f}")
    return model


if __name__ == "__main__":
    app()
