from collections import Counter

import pandas as pd
import pytest

from ensembles.decision_tree import DecisionTree, LeafNode, build_tree
from ensembles.random_forest import (
    PerformanceMetrics,
    RandomForest,
    RandomForestTrainingData,
    build_forest,
    format_results_table,
    parse_int_range,
    render_progress,
    split_data,
    train_forest,
)


@pytest.fixture
def data():
    return RandomForestTrainingData.create()


@pytest.fixture
def split(data):
    return split_data(data, train_fraction=0.8, seed=42)


@pytest.fixture
def forest(split):
    train_data, _ = split
    return build_forest(train_data, num_trees=5, seed=42, max_depth=3)


# --- RandomForestTrainingData ---


def test_create_returns_training_data(data):
    assert isinstance(data, RandomForestTrainingData)
    assert len(data.x) > 0
    assert len(data.y) == len(data.x)
    assert len(data.features) > 0


def test_bootstrapped_same_size(data):
    bootstrapped = data.bootstrapped(seed=0)
    assert len(bootstrapped.x) == len(data.x)
    assert len(bootstrapped.y) == len(data.x)
    assert bootstrapped.features == data.features


def test_bootstrapped_contains_duplicates_on_average(data):
    bootstrapped = data.bootstrapped(seed=0)
    unique_rows = bootstrapped.x.drop_duplicates().shape[0]
    assert unique_rows < len(bootstrapped.x)


def test_bootstrapped_seed_reproducible(data):
    b1 = data.bootstrapped(seed=7)
    b2 = data.bootstrapped(seed=7)
    pd.testing.assert_frame_equal(b1.x, b2.x)


# --- split_data ---


def test_split_data_proportions(data):
    train, test = split_data(data, train_fraction=0.8, seed=0)
    total = len(data.x)
    assert len(train.x) == int(0.8 * total)
    assert len(test.x) == total - int(0.8 * total)


def test_split_data_no_overlap(data):
    train, test = split_data(data, train_fraction=0.8, seed=0)
    assert len(train.x) + len(test.x) == len(data.x)


def test_split_data_features_preserved(data):
    train, test = split_data(data, train_fraction=0.8, seed=0)
    assert train.features == data.features
    assert test.features == data.features


# --- build_forest ---


def test_build_forest_correct_num_trees(split):
    train_data, _ = split
    forest = build_forest(train_data, num_trees=7, seed=42, max_depth=3)
    assert len(forest.trees) == 7


def test_build_forest_trees_are_decision_trees(split):
    train_data, _ = split
    forest = build_forest(train_data, num_trees=3, seed=0, max_depth=3)
    for tree in forest.trees:
        assert isinstance(tree, DecisionTree)


# --- RandomForest.predict_votes and predict ---


def test_predict_votes_returns_list(forest, split):
    _, test_data = split
    row = test_data.x.iloc[0]
    votes = forest.predict_votes(row)
    assert isinstance(votes, list)
    assert len(votes) == len(forest.trees)
    assert all(v in [0, 1] for v in votes)


def test_predict_returns_majority_vote(forest, split):
    _, test_data = split
    row = test_data.x.iloc[0]
    prediction = forest.predict(row)
    votes = forest.predict_votes(row)
    expected = Counter(votes).most_common(1)[0][0]
    assert prediction == expected


def test_predict_valid_labels(forest, split):
    _, test_data = split
    for i in range(min(10, len(test_data.x))):
        row = test_data.x.iloc[i]
        assert forest.predict(row) in [0, 1]


# --- RandomForest.evaluate ---


def test_evaluate_returns_performance_metrics(forest, split):
    _, test_data = split
    metrics = forest.evaluate(test_data.x, test_data.y)
    assert isinstance(metrics, PerformanceMetrics)


def test_evaluate_metrics_in_range(forest, split):
    _, test_data = split
    metrics = forest.evaluate(test_data.x, test_data.y)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1_score <= 1


def test_evaluate_confusion_matrix_keys(forest, split):
    _, test_data = split
    metrics = forest.evaluate(test_data.x, test_data.y)
    assert "true_positives" in metrics.confusion_matrix
    assert "false_positives" in metrics.confusion_matrix
    assert "true_negatives" in metrics.confusion_matrix
    assert "false_negatives" in metrics.confusion_matrix


def test_evaluate_accuracy_above_random(forest, split):
    _, test_data = split
    metrics = forest.evaluate(test_data.x, test_data.y)
    assert metrics.accuracy > 0.5


# --- RandomForest.feature_usage ---


def test_feature_usage_returns_counts(forest):
    counts, total = forest.feature_usage()
    assert total >= 0
    assert all(isinstance(k, str) for k in counts)
    assert all(isinstance(v, int) for v in counts.values())


def test_feature_usage_total_matches_sum(forest):
    counts, total = forest.feature_usage()
    assert sum(counts.values()) == total


# --- RandomForest.vote_report ---


def test_vote_report_length(forest, split):
    _, test_data = split
    report = forest.vote_report(test_data.x, test_data.y)
    assert len(report) == len(test_data.x)


def test_vote_report_fields(forest, split):
    _, test_data = split
    report = forest.vote_report(test_data.x, test_data.y)
    for entry in report[:5]:
        assert "index" in entry
        assert "prediction" in entry
        assert "true_label" in entry
        assert "vote_counts" in entry


# --- train_forest ---


def test_train_forest_returns_forest_and_metrics(split):
    train_data, test_data = split
    forest, metrics = train_forest(train_data, test_data, num_trees=5, seed=0, max_depth=3)
    assert isinstance(forest, RandomForest)
    assert isinstance(metrics, PerformanceMetrics)


# --- parse_int_range ---


def test_parse_int_range_basic():
    assert parse_int_range("1:5", "test") == [1, 2, 3, 4, 5]


def test_parse_int_range_with_step():
    assert parse_int_range("10:50:10", "test") == [10, 20, 30, 40, 50]


def test_parse_int_range_single_value():
    assert parse_int_range("3:3", "test") == [3]


def test_parse_int_range_invalid_format():
    with pytest.raises(ValueError, match="format"):
        parse_int_range("5", "test")


def test_parse_int_range_step_zero():
    with pytest.raises(ValueError, match="step"):
        parse_int_range("1:5:0", "test")


def test_parse_int_range_end_less_than_start():
    with pytest.raises(ValueError, match="end must be >= start"):
        parse_int_range("5:3", "test")


def test_parse_int_range_below_min():
    with pytest.raises(ValueError, match="start must be"):
        parse_int_range("0:5", "test", min_value=1)


def test_parse_int_range_above_max():
    with pytest.raises(ValueError, match="end must be <="):
        parse_int_range("1:10", "test", max_value=5)


# --- format_results_table ---


def test_format_results_table_contains_headers():
    metrics = PerformanceMetrics(
        accuracy=0.8,
        precision=0.75,
        recall=0.7,
        f1_score=0.72,
        confusion_matrix={"true_positives": 10, "false_positives": 2, "true_negatives": 8, "false_negatives": 3},
    )
    results = [(10, 3, metrics, None)]
    table = format_results_table(results, top_n=1)
    assert "num_trees" in table
    assert "accuracy" in table
    assert "f1" in table


def test_format_results_table_top_n():
    metrics = PerformanceMetrics(
        accuracy=0.8,
        precision=0.75,
        recall=0.7,
        f1_score=0.72,
        confusion_matrix={"true_positives": 10, "false_positives": 2, "true_negatives": 8, "false_negatives": 3},
    )
    results = [(i, 3, metrics, None) for i in range(10, 60, 10)]
    table = format_results_table(results, top_n=3)
    lines = table.strip().split("\n")
    assert len(lines) == 5  # header + separator + 3 data rows


# --- render_progress ---


def test_render_progress_full():
    result = render_progress(30, 30)
    assert "30/30" in result
    assert "#" in result
    assert result.startswith("[")
    assert "]" in result
    assert "-" not in result.split("]")[0]


def test_render_progress_zero():
    result = render_progress(0, 30)
    assert "0/30" in result


def test_render_progress_empty_total():
    result = render_progress(0, 0)
    assert result == ""
