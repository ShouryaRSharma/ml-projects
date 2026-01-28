from dataclasses import dataclass
import pandas as pd


@dataclass
class TrainingData:
    x: pd.DataFrame | pd.Series
    y: pd.Series


@dataclass
class ModelState:
    w: float
    b: float
    lr: float
    iterations: int
