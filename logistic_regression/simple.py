from dataclasses import dataclass, field
from typing import Any, Self

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
import typer
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

matplotlib.use("TkAgg")
app = typer.Typer()


@dataclass
class TrainingData:
    x: np.ndarray
    y: np.ndarray
    features: list[str]
    target_names: list[str]


@dataclass
class ModelState:
    w: np.ndarray
    b: float
    lr: float
    iterations: int

    def predict(self: Self, x: np.ndarray) -> np.ndarray:
        z = np.dot(x, self.w) + self.b
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def step(self: Self, x: np.ndarray, y: np.ndarray) -> np.floating[Any]:
        probabilities = self.predict(x)
        error = y - probabilities

        derivative_b = -1 * error.mean()
        derivative_w = -1 * np.dot(x.T, error) / len(y)
        self.b -= self.lr * derivative_b
        self.w -= self.lr * derivative_w
        return self.loss(x, y)

    def loss(self, x: np.ndarray, y: np.ndarray) -> np.floating[Any]:
        probs = self.predict(x)
        epsilon = 1e-15  # Avoid log(0)
        loss = -np.mean(
            y * np.log(probs + epsilon) + (1 - y) * np.log(1 - probs + epsilon)
        )
        return loss


@dataclass
class Visualizer:
    fig: Figure
    ax: Axes
    line: Line2D
    losses: list[np.floating[Any]] = field(default_factory=list)
    iters: list[int] = field(default_factory=list)


def init_visualizer() -> Visualizer:
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    (line,) = ax.plot([], [], color="blue", label="Log Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress: Log Loss")
    ax.grid(True, alpha=0.3)
    return Visualizer(fig=fig, ax=ax, line=line)


def fetch_data() -> TrainingData:
    dataset = sklearn.datasets.load_breast_cancer()
    X: np.ndarray = dataset.data  # type: ignore[reportUnknownVariableType]
    X: np.ndarray = (X - X.mean(axis=0)) / X.std(axis=0)

    return TrainingData(
        x=X,
        y=dataset.target,
        features=dataset.feature_names.tolist(),
        target_names=dataset.target_names.tolist(),
    )


def optimise(data: TrainingData, model: ModelState, viz: Visualizer):
    for i in range(model.iterations):
        current_loss = model.step(data.x, data.y)

        if i % 10 == 0:
            viz.losses.append(current_loss)
            viz.iters.append(i)

            # Update plot
            viz.line.set_data(viz.iters, viz.losses)
            viz.ax.relim()
            viz.ax.autoscale_view()
            viz.fig.canvas.draw_idle()
            viz.fig.canvas.flush_events()
            plt.pause(0.001)


@app.command()
def train(
    iterations: int = typer.Option(500, help="Steps of gradient descent"),
    lr: float = typer.Option(0.1, help="Step size (learning rate)"),
):
    data = fetch_data()
    model = ModelState(
        w=np.zeros(len(data.features)), b=0.0, lr=lr, iterations=iterations
    )

    viz = init_visualizer()
    typer.echo(f"Training on {len(data.features)} features...")

    optimise(data, model, viz)

    final_probs = model.predict(data.x)
    preds = (final_probs >= 0.5).astype(int)
    accuracy = (preds == data.y).mean()

    typer.echo("\n" + "=" * 30)
    typer.echo(f"Final Accuracy: {accuracy:.2%}")
    typer.echo(f"Final Loss:     {viz.losses[-1]:.4f}")
    typer.echo("=" * 30)

    importance = pd.Series(model.w, index=data.features).sort_values(ascending=False)

    typer.echo("\n--- Top Coefficients (ML Weights) ---")
    typer.echo(importance.head(5))
    typer.echo("\n--- Bottom Coefficients ---")
    typer.echo(importance.tail(5))

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    app()
