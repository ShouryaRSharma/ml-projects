from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import random
from typing import Self, TypedDict

from typer import Typer

import pandas as pd

from ensembles.decision_tree import (
    DecisionNode,
    DecisionTree,
    LeafNode,
    Node,
    TrainingData,
    build_tree,
)

app = Typer(name="random_forest")

_WORKER_TRAIN_DATA: RandomForestTrainingData | None = None
_WORKER_TEST_DATA: RandomForestTrainingData | None = None
_WORKER_DATASET: str | None = None
_WORKER_TRAIN_FRACTION: float | None = None
_WORKER_SPLIT_SEED: int | None = None


@dataclass
class RandomForestTrainingData(TrainingData):
    @classmethod
    def create(cls, dataset: str = "titanic") -> "RandomForestTrainingData":
        base = TrainingData.create(dataset=dataset)
        return cls(x=base.x, y=base.y, features=base.features)

    def bootstrapped(self: Self, seed: int | None = None) -> TrainingData:
        indices = self.x.sample(frac=1, replace=True, random_state=seed).index

        return TrainingData(
            x=self.x.loc[indices, self.features].reset_index(drop=True),
            y=self.y.loc[indices].reset_index(drop=True),
            features=self.features,
        )


@dataclass
class PerformanceMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: dict[str, int]


class VoteReport(TypedDict):
    index: int
    prediction: int
    true_label: int
    vote_counts: dict[int, int]


@dataclass
class RandomForest:
    trees: list[DecisionTree]

    def predict_votes(self, x: pd.Series) -> list[int]:
        return [tree.predict(x) for tree in self.trees]

    def predict(self, x: pd.Series) -> int:
        votes = self.predict_votes(x)
        counts = Counter(votes)
        return counts.most_common(1)[0][0]

    def evaluate(self, x: pd.DataFrame, y: pd.Series) -> PerformanceMetrics:
        predictions = [self.predict(row) for _, row in x.iterrows()]
        predictions_series = pd.Series(predictions, index=y.index)

        correct = (predictions_series == y).sum()
        total = len(y)
        accuracy = correct / total

        tp = ((predictions_series == 1) & (y == 1)).sum()
        fp = ((predictions_series == 1) & (y == 0)).sum()
        fn = ((predictions_series == 0) & (y == 1)).sum()
        tn = ((predictions_series == 0) & (y == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix={
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            },
        )

    def feature_usage(self) -> tuple[Counter[str], int]:
        counts: Counter[str] = Counter()

        def _collect(node: Node) -> None:
            if isinstance(node, DecisionNode):
                counts[node.feature] += 1
                _collect(node.left_child)
                _collect(node.right_child)

        for tree in self.trees:
            _collect(tree.root)

        total = sum(counts.values())
        return counts, total

    def vote_report(self, x: pd.DataFrame, y: pd.Series) -> list[VoteReport]:
        report: list[VoteReport] = []
        for i in range(len(x)):
            row = x.iloc[i]
            votes = self.predict_votes(row)
            counts = Counter(votes)
            prediction = counts.most_common(1)[0][0]
            report.append(
                {
                    "index": i,
                    "prediction": int(prediction),
                    "true_label": int(y.iloc[i]),
                    "vote_counts": dict(sorted(counts.items())),
                }
            )
        return report


def build_forest(
    dataset: RandomForestTrainingData,
    num_trees: int,
    num_features: int | None = None,
    seed: int | None = None,
    max_depth: int = 5,
) -> RandomForest:
    trees: list[DecisionTree] = []
    for i in range(num_trees):  # Number of trees
        tree_seed = None if seed is None else seed + i
        bootstrap = dataset.bootstrapped(seed=tree_seed)
        tree_rng = random.Random(tree_seed) if tree_seed is not None else None
        tree = build_tree(
            data=bootstrap,
            max_depth=max_depth,
            current_depth=0,
            num_features=num_features,
            rng=tree_rng,
        )
        trees.append(DecisionTree(root=tree))

    forest = RandomForest(trees=trees)
    return forest


def split_data(
    dataset: RandomForestTrainingData,
    train_fraction: float = 0.8,
    seed: int | None = None,
) -> tuple[RandomForestTrainingData, RandomForestTrainingData]:
    indices = dataset.x.sample(frac=1, random_state=seed).index
    x_shuffled = dataset.x.loc[indices].reset_index(drop=True)
    y_shuffled = dataset.y.loc[indices].reset_index(drop=True)

    train_size = int(train_fraction * len(x_shuffled))
    train_data = RandomForestTrainingData(
        x=x_shuffled.iloc[:train_size].reset_index(drop=True),
        y=y_shuffled.iloc[:train_size].reset_index(drop=True),
        features=dataset.features,
    )
    test_data = RandomForestTrainingData(
        x=x_shuffled.iloc[train_size:].reset_index(drop=True),
        y=y_shuffled.iloc[train_size:].reset_index(drop=True),
        features=dataset.features,
    )
    return train_data, test_data


def _init_worker(dataset: str, train_fraction: float, split_seed: int | None) -> None:
    global _WORKER_TRAIN_DATA
    global _WORKER_TEST_DATA
    global _WORKER_DATASET
    global _WORKER_TRAIN_FRACTION
    global _WORKER_SPLIT_SEED

    data = RandomForestTrainingData.create(dataset=dataset)
    train_data, test_data = split_data(
        data, train_fraction=train_fraction, seed=split_seed
    )
    _WORKER_TRAIN_DATA = train_data
    _WORKER_TEST_DATA = test_data
    _WORKER_DATASET = dataset
    _WORKER_TRAIN_FRACTION = train_fraction
    _WORKER_SPLIT_SEED = split_seed


def _get_worker_split(
    dataset: str, train_fraction: float, split_seed: int | None
) -> tuple[RandomForestTrainingData, RandomForestTrainingData]:
    global _WORKER_TRAIN_DATA
    global _WORKER_TEST_DATA
    global _WORKER_DATASET
    global _WORKER_TRAIN_FRACTION
    global _WORKER_SPLIT_SEED

    if (
        _WORKER_TRAIN_DATA is not None
        and _WORKER_TEST_DATA is not None
        and _WORKER_DATASET == dataset
        and _WORKER_TRAIN_FRACTION == train_fraction
        and _WORKER_SPLIT_SEED == split_seed
    ):
        return _WORKER_TRAIN_DATA, _WORKER_TEST_DATA

    data = RandomForestTrainingData.create(dataset=dataset)
    train_data, test_data = split_data(
        data, train_fraction=train_fraction, seed=split_seed
    )
    _WORKER_TRAIN_DATA = train_data
    _WORKER_TEST_DATA = test_data
    _WORKER_DATASET = dataset
    _WORKER_TRAIN_FRACTION = train_fraction
    _WORKER_SPLIT_SEED = split_seed
    return train_data, test_data


def train_forest(
    train_data: RandomForestTrainingData,
    test_data: RandomForestTrainingData,
    num_trees: int,
    num_features: int | None = None,
    seed: int | None = None,
    max_depth: int = 5,
) -> tuple[RandomForest, PerformanceMetrics]:
    forest = build_forest(
        train_data,
        num_trees,
        num_features=num_features,
        seed=seed,
        max_depth=max_depth,
    )  # type: ignore
    forest_metrics = forest.evaluate(test_data.x, test_data.y)
    return forest, forest_metrics


def parse_int_range(
    spec: str,
    name: str,
    min_value: int = 1,
    max_value: int | None = None,
) -> list[int]:
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"{name} must be in 'start:end[:step]' format, got '{spec}'")

    start = int(parts[0])
    end = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else 1

    if step <= 0:
        raise ValueError(f"{name} step must be > 0, got {step}")
    if start < min_value:
        raise ValueError(f"{name} start must be >= {min_value}, got {start}")
    if end < start:
        raise ValueError(f"{name} end must be >= start, got {start}:{end}")
    if max_value is not None and end > max_value:
        raise ValueError(f"{name} end must be <= {max_value}, got {end}")

    values = list(range(start, end + 1, step))
    if not values:
        raise ValueError(f"{name} range produced no values from '{spec}'")
    return values


def evaluate_combo(
    num_trees: int,
    num_features: int,
    seed: int | None,
    top_features: int = 0,
    dataset: str = "titanic",
    train_fraction: float = 0.8,
    split_seed: int | None = None,
    max_depth: int = 5,
) -> tuple[int, int, PerformanceMetrics, list[str] | None]:
    train_data, test_data = _get_worker_split(
        dataset, train_fraction=train_fraction, split_seed=split_seed
    )
    forest, metrics = train_forest(
        train_data,
        test_data,
        num_trees=num_trees,
        num_features=num_features,
        seed=seed,
        max_depth=max_depth,
    )
    summary: list[str] | None = None
    if top_features > 0:
        usage, total = forest.feature_usage()
        if total > 0:
            summary = [
                f"{feature}:{count / total:.1%}"
                for feature, count in usage.most_common(top_features)
            ]
        else:
            summary = []
    return num_trees, num_features, metrics, summary


def format_results_table(
    results: list[tuple[int, int, PerformanceMetrics, list[str] | None]],
    top_n: int = 5,
) -> str:
    include_summary = any(result[3] for result in results)
    headers = ["rank", "num_trees", "num_features"]
    if include_summary:
        headers.append("top_features")
    headers += ["accuracy", "precision", "recall", "f1"]

    rows: list[list[str]] = []
    for rank, (trees, features, metrics, feature_subset) in enumerate(
        results[:top_n], start=1
    ):
        row = [str(rank), str(trees), str(features)]
        if include_summary:
            row.append(",".join(feature_subset) if feature_subset else "-")
        row += [
            f"{metrics.accuracy:.4f}",
            f"{metrics.precision:.4f}",
            f"{metrics.recall:.4f}",
            f"{metrics.f1_score:.4f}",
        ]
        rows.append(row)

    columns = [headers] + rows
    widths = [max(len(row[i]) for row in columns) for i in range(len(headers))]

    lines = [
        " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))),
        "-+-".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return ""
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def print_progress(current: int, total: int) -> None:
    print(f"\r{render_progress(current, total)}", end="", flush=True)
    if current >= total:
        print()


@app.command()
def train(
    num_trees: int = 20,
    num_features: int | None = None,
    debug: bool = False,
    show_disagreements: int = 0,
    seed: int | None = None,
    dataset: str = "titanic",
    max_depth: int = 5,
) -> PerformanceMetrics:
    data = RandomForestTrainingData.create(dataset=dataset)
    train_data, test_data = split_data(data, seed=seed)

    forest, forest_metrics = train_forest(
        train_data,
        test_data,
        num_trees=num_trees,
        num_features=num_features,
        seed=seed,
        max_depth=max_depth,
    )
    print(f"Random Forest Performance: {forest_metrics.__dict__}")

    if debug or show_disagreements > 0:
        report = forest.vote_report(test_data.x, test_data.y)
        non_unanimous = [entry for entry in report if len(entry["vote_counts"]) > 1]
        misclassified = [
            entry for entry in report if entry["prediction"] != entry["true_label"]
        ]

        if debug:
            print(
                "Vote summary:"
                f" non-unanimous={len(non_unanimous)}/{len(report)}"
                f", misclassified={len(misclassified)}/{len(report)}"
            )

        if show_disagreements > 0:
            print("Sample disagreements (non-unanimous votes):")
            for entry in non_unanimous[:show_disagreements]:
                print(
                    "index="
                    f"{entry['index']}"
                    " true="
                    f"{entry['true_label']}"
                    " pred="
                    f"{entry['prediction']}"
                    " votes="
                    f"{entry['vote_counts']}"
                )
    return forest_metrics


@app.command()
def tune(
    trees_range: str = "10:50:10",
    features_range: str | None = None,
    seed: int | None = None,
    max_workers: int | None = None,
    progress: bool = True,
    top_features: int = 3,
    dataset: str = "titanic",
    train_fraction: float = 0.8,
    max_depth: int = 5,
) -> None:
    data = RandomForestTrainingData.create(dataset=dataset)
    total_features = len(data.features)
    trees_grid = parse_int_range(trees_range, "trees_range", min_value=1)
    if features_range is None:
        features_range = f"1:{total_features}:1"
    feature_grid = parse_int_range(
        features_range, "features_range", min_value=1, max_value=total_features
    )

    combos: list[tuple[int, int, int | None, str]] = []
    combo_index = 0
    for trees in trees_grid:
        for features in feature_grid:
            combo_seed = None if seed is None else seed + combo_index
            combos.append((trees, features, combo_seed, dataset))
            combo_index += 1

    results: list[tuple[int, int, PerformanceMetrics, list[str] | None]] = []
    total_jobs = len(combos)
    if progress:
        print_progress(0, total_jobs)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(dataset, train_fraction, seed),
    ) as executor:
        future_map = {
            executor.submit(
                evaluate_combo,
                trees,
                features,
                combo_seed,
                top_features,
                dataset_name,
                train_fraction,
                seed,
                max_depth,
            ): (trees, features)
            for trees, features, combo_seed, dataset_name in combos
        }
        completed = 0
        for future in as_completed(future_map):
            results.append(future.result())
            completed += 1
            if progress:
                print_progress(completed, total_jobs)

    ranked = sorted(results, key=lambda item: item[2].f1_score, reverse=True)
    print("Top 5 by f1:")
    print(format_results_table(ranked, top_n=5))


if __name__ == "__main__":
    app()
