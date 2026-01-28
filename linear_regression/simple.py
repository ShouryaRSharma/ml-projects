from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets
import typer
from linear_regression.common import ModelState, TrainingData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

matplotlib.use("TkAgg")
app = typer.Typer()


@dataclass
class Visualizer:
    fig: Figure
    ax: Axes
    line: Line2D


def fetch_data() -> TrainingData:
    dataset = sklearn.datasets.fetch_california_housing(as_frame=True)
    return TrainingData(x=dataset.frame["MedInc"], y=dataset.frame["MedHouseVal"])


def init_visualizer(data: TrainingData, model: ModelState) -> Visualizer:
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data.x, data.y, alpha=0.2, color="gray", s=10)
    (line,) = ax.plot(data.x, model.w * data.x + model.b, color="red", linewidth=2)
    ax.set_xlabel("Median Income")
    ax.set_ylabel("House Value")

    plt.show(block=False)
    fig.canvas.draw()

    return Visualizer(fig=fig, ax=ax, line=line)


def update_ui(
    viz: Visualizer, data: TrainingData, model: ModelState, iteration: int
) -> None:
    # y = wx + b
    viz.line.set_ydata(model.w * data.x + model.b)
    viz.ax.set_title(f"Iteration: {iteration} | W: {model.w:.3f} | B: {model.b:.3f}")

    viz.fig.canvas.draw_idle()
    viz.fig.canvas.flush_events()
    plt.pause(0.01)


def optimise(data: TrainingData, model: ModelState, viz: Visualizer) -> ModelState:
    for i in range(model.iterations):
        # Calculate current predictions and error
        y_pred = model.w * data.x + model.b
        error = data.y - y_pred

        # Gradient Descent
        derivative_b = -2 * error.mean()
        derivative_w = -2 * (data.x * error).mean()

        model.b -= model.lr * derivative_b
        model.w -= model.lr * derivative_w

        if i % 10 == 0:
            update_ui(viz, data, model, i)

    return model


@app.command()
def train(iterations: int = 500, lr: float = 0.01, w: float = 0.0, b: float = 0.0):
    data = fetch_data()
    model = ModelState(w=w, b=b, lr=lr, iterations=iterations)
    viz = init_visualizer(data, model)

    print(f"Starting: W={model.w}, B={model.b}, LR={model.lr}")

    final_state = optimise(data, model, viz)

    print(f"Final results -> W: {final_state.w:.4f}, B: {final_state.b:.4f}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    app()
