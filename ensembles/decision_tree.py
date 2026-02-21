from __future__ import annotations

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
import random
from typing import cast

import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import typer

app = typer.Typer()


@dataclass
class TrainingData:
    x: pd.DataFrame
    y: pd.Series
    features: list[str]

    @classmethod
    def create(cls, dataset: str = "titanic") -> "TrainingData":
        normalized = dataset.strip().lower()
        if normalized == "titanic":
            df = sns.load_dataset("titanic")
            df = df.dropna()
            df["sex"] = df["sex"].replace({"male": 0, "female": 1})

            features = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
            x = pd.DataFrame(df[features])
            y = pd.Series(df["survived"])
            return TrainingData(x=x, y=y, features=features)

        if normalized in {"breast_cancer", "breast-cancer", "cancer"}:
            bunch = load_breast_cancer(as_frame=True)
            x = pd.DataFrame(bunch.data)  # type: ignore[arg-type]
            y = pd.Series(bunch.target)  # type: ignore[arg-type]
            features = list(x.columns)
            return TrainingData(x=x, y=y, features=features)

        raise ValueError("dataset must be 'titanic' or 'breast_cancer'")


@dataclass
class DecisionTree:
    root: Node

    def predict(self, sample: pd.Series) -> int:
        return self.root.predict(sample)

    def evaluate(self, x: pd.DataFrame, y: pd.Series) -> dict:
        predictions = []
        for i in range(len(x)):
            sample = x.iloc[i]
            predictions.append(self.root.predict(sample))

        predictions = pd.Series(predictions, index=y.index)

        correct = (predictions == y).sum()
        total = len(y)
        accuracy = correct / total

        # Calculate precision, recall for binary classification
        tp = ((predictions == 1) & (y == 1)).sum()
        fp = ((predictions == 1) & (y == 0)).sum()
        fn = ((predictions == 0) & (y == 1)).sum()
        tn = ((predictions == 0) & (y == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            },
        }

    def print_tree(self) -> None:
        def _print_recursive(node: Node, prefix: str, is_last: bool):
            connector = "└── " if is_last else "├── "

            if isinstance(node, LeafNode):
                node_info = f"Predict: {node.majority_class} (samples={node.samples}, gini={node.gini_impurity:.3f})"
            else:
                decision_node = cast(DecisionNode, node)
                node_info = (
                    f"[ {decision_node.feature} <= {decision_node.threshold:.2f} ] "
                    f"(samples={decision_node.samples}, gini={decision_node.gini_impurity:.3f})"
                )

            print(f"{prefix}{connector}{node_info}")

            child_prefix = prefix + ("    " if is_last else "│   ")

            if not isinstance(node, LeafNode):
                decision_node = cast(DecisionNode, node)
                _print_recursive(decision_node.left_child, child_prefix, is_last=False)
                _print_recursive(decision_node.right_child, child_prefix, is_last=True)

        if self.root:
            _print_recursive(self.root, "", True)


@dataclass
class Node:
    gini_impurity: float
    samples: int

    @abstractmethod
    def predict(self, sample: pd.Series) -> int:
        pass


@dataclass
class DecisionNode(Node):
    feature: str
    left_child: Node
    right_child: Node
    threshold: float

    def predict(self, sample: pd.Series) -> int:
        if sample[self.feature] <= self.threshold:
            return self.left_child.predict(sample)
        else:
            return self.right_child.predict(sample)


@dataclass
class LeafNode(Node):
    majority_class: int

    def predict(self, sample: pd.Series) -> int:
        return self.majority_class


def gini_impurity(y: pd.Series) -> float:
    if len(y) == 0:
        return 0.0

    probs = y.value_counts(normalize=True)

    return 1.0 - (probs**2).sum()


def create_leaf_node(y: pd.Series) -> LeafNode:
    majority_class = int(y.value_counts().idxmax())  # type: ignore[arg-type]
    impurity = gini_impurity(y)
    samples = len(y)

    return LeafNode(
        gini_impurity=impurity,
        samples=samples,
        majority_class=majority_class,
    )


def find_best_split(
    x: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    num_features: int | None = None,
    rng: random.Random | None = None,
) -> tuple[str | None, float | None]:
    best_gini = float("inf")
    best_feature = None
    best_threshold = None

    features_to_consider = features
    if num_features is not None:
        total_features = len(features)
        if num_features < 1 or num_features > total_features:
            raise ValueError(
                "num_features must be between 1 and "
                f"{total_features}, got {num_features}"
            )
        sampler = rng if rng is not None else random
        features_to_consider = sampler.sample(features, k=num_features)

    for feature in features_to_consider:
        thresholds = x[feature].unique()
        for threshold in thresholds:
            left_mask = x[feature] <= threshold
            right_mask = x[feature] > threshold

            y_left = pd.Series(y[left_mask])
            if len(y_left) == 0 or len(y[left_mask]) == len(y):
                continue  # Skip if split doesn't separate samples (all on one side or empty split)

            y_right = pd.Series(y[right_mask])

            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)

            weighted_fini_left = (len(y_left) / len(y)) * gini_left
            weighted_gini_right = (len(y_right) / len(y)) * gini_right
            weighted_gini = weighted_fini_left + weighted_gini_right

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


def build_tree(
    data: TrainingData,
    max_depth: int,
    current_depth: int,
    num_features: int | None = None,
    rng: random.Random | None = None,
) -> Node:
    feature, threshold = find_best_split(
        data.x, data.y, data.features, num_features=num_features, rng=rng
    )
    if (
        len(data.y.unique()) == 1
        or feature is None
        or threshold is None
        or current_depth >= max_depth
    ):
        return create_leaf_node(data.y)

    threshold_value = cast(float, threshold)

    left_mask = data.x[feature] <= threshold_value
    right_mask = data.x[feature] > threshold_value
    build_left = build_tree(
        data=TrainingData(
            x=pd.DataFrame(data.x[left_mask]),
            y=pd.Series(data.y[left_mask]),
            features=data.features,
        ),
        max_depth=max_depth,
        current_depth=current_depth + 1,
        num_features=num_features,
        rng=rng,
    )
    build_right = build_tree(
        data=TrainingData(
            x=pd.DataFrame(data.x[right_mask]),
            y=pd.Series(data.y[right_mask]),
            features=data.features,
        ),
        max_depth=max_depth,
        current_depth=current_depth + 1,
        num_features=num_features,
        rng=rng,
    )

    return DecisionNode(
        feature=feature,
        gini_impurity=gini_impurity(data.y),
        samples=len(data.y),
        left_child=build_left,
        right_child=build_right,
        threshold=threshold_value,
    )
