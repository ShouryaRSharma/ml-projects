# 🚀 From Calculus to Transformers: A Machine Learning Journey

This project is a structured series of experiments designed to demystify how AI actually learns. Instead of just calling library functions, we build the foundations from scratch to understand the optimization, math, and architecture that powers modern intelligence.

---

## 🛠️ The Roadmap

| Module | Dataset | Model Type | Implementation Level |
| :--- | :--- | :--- | :--- |
| **01. Linear Regression** | California Housing | Regression | **Scratch** (Manual Gradient Descent) |
| **02. Logistic Regression** | Breast Cancer | Classification | **Scratch** (Sigmoid + Log-Loss) |
| **03. Random Forests** | Titanic Survival | Ensemble | **Mixed** (Scratch Tree + XGBoost Library) |
| **04. Deep Learning** | MNIST Digits | Neural Network | **PyTorch** (Multi-Layer Perceptron) |
| **05. Transformers** | IMDb Reviews | Attention | **PyTorch** (Attention Mechanism) |

---

## 🟢 Module 1: Linear Regression & The "Engine"

**Goal:** Find the best-fit line ($y = Wx + B$) for housing prices.

* **The Math:** Manually implement Partial Derivatives of Mean Squared Error (MSE).
* **The Key Learning:** Gradient Descent. Understanding that the "Gradient" is just a compass pointing "up the hill," and we move the opposite way.
* **The Implementation:** * Calculate `error = predicted - actual`.
  * `derivative_w = 2/n * sum(error * x)`.
  * `derivative_b = 2/n * sum(error)`.

---

## 🟡 Module 2: Logistic Regression & Classification

**Goal:** Predict if a tumor is Malignant or Benign.

* **The Math:** Squashing the linear output through the **Sigmoid Function** $1 / (1 + e^{-z})$.
* **The Key Learning:** Cross-Entropy Loss. Why Squared Error doesn't work for "Yes/No" questions.

---

## 🟠 Module 3: Decision Trees & Boosting

**Goal:** Predict Titanic survival using non-linear logic.

* **The Math:** **Gini Impurity** and **Entropy**. How a model decides the "best split" (e.g., `Age > 10`).
* **The Key Learning:** The difference between **Bagging** (Random Forest) and **Boosting** (XGBoost).
* **Implementation:** Build a single decision tree from scratch, then use the `XGBoost` library for performance.

---

## 🔴 Module 4: Deep Learning (PyTorch)

**Goal:** Classify handwritten digits (MNIST).

* **The Transition:** Move from manual NumPy math to **PyTorch Tensors**.
* **The Key Learning:** **Backpropagation**. Using `loss.backward()` to let the computer handle the calculus of many layers.
* **Architecture:** Input Layer (784) → Hidden Layer (ReLU) → Output Layer (10 Softmax).

---

## 🟣 Module 5: Transformers & Attention (PyTorch)

**Goal:** Sentiment analysis on movie reviews.

* **The Key Learning:** **Self-Attention**. Why "not" in "not good" needs to be linked to "good."
* **The Transition:** Skip RNNs/LSTMs. Build the **Attention Mechanism** ($Q, K, V$ matrices) directly in PyTorch.
* **Architecture:** Use `nn.TransformerEncoder` to process text sequences in parallel.

---

## 📚 Essential Watchlist

* **StatQuest:** Linear Regression & Gradient Descent.
* **StatQuest:** Logistic Regression & Odds.
* **3Blue1Brown:** Neural Networks & Backpropagation.
* **Jay Alammar:** The Illustrated Transformer.
