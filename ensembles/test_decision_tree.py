import pytest

from ensembles.decision_tree import DecisionTree, TrainingData, build_tree


@pytest.fixture
def data():
    return TrainingData.create()


@pytest.fixture
def tree(data):
    return DecisionTree(root=build_tree(data, max_depth=3, current_depth=0))


def test_tree_structure_is_valid(tree):
    assert tree.root is not None
    assert hasattr(tree.root, "predict")


def test_predictions_are_valid(tree, data):
    for i in range(min(10, len(data.x))):
        sample = data.x.iloc[i]
        prediction = tree.root.predict(sample)
        assert prediction in [0, 1], f"Prediction {prediction} should be 0 or 1"


def test_train_test_split_performance(data):
    train_size = int(0.8 * len(data.x))
    train_data = TrainingData(
        x=data.x.iloc[:train_size], y=data.y.iloc[:train_size], features=data.features
    )
    test_data = TrainingData(
        x=data.x.iloc[train_size:], y=data.y.iloc[train_size:], features=data.features
    )

    train_tree = DecisionTree(root=build_tree(train_data, max_depth=3, current_depth=0))

    correct_predictions = 0
    for i in range(len(test_data.x)):
        sample = test_data.x.iloc[i]
        prediction = train_tree.root.predict(sample)
        actual = test_data.y.iloc[i]
        if prediction == actual:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data.x)
    assert accuracy > 0.5, (
        f"Accuracy {accuracy:.2f} should be better than random guessing"
    )


def test_leaf_node_creation(data):
    simple_data = TrainingData(
        x=data.x.iloc[:5], y=data.y.iloc[:5], features=data.features
    )
    leaf_tree = DecisionTree(root=build_tree(simple_data, max_depth=1, current_depth=0))
    predictions = [
        leaf_tree.root.predict(simple_data.x.iloc[i]) for i in range(len(simple_data.x))
    ]
    assert all(p in [0, 1] for p in predictions)
