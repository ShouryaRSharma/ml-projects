# Module 3: Random Forests & Boosting (Ensembles)

**The Project:** Titanic Survival Predictor
**The Goal:** Predict survival by building a "Forest" of logic, then compare it to a high-performance Gradient Boosting library (XGBoost).

## 🎯 Learning Objectives

* **Information Gain/Gini:** The math used to decide where to "cut" the data (e.g., Is Age > 10?).
* **Bootstrapping (Bagging):** Learning how to randomly sample your data so every tree sees a different "version" of the truth.
* **Majority Voting:** How the forest aggregates 100 different tree opinions into one final answer.
* **Boosting vs. Bagging:** Understanding why Random Forest (Bagging) builds trees in parallel, while XGBoost (Boosting) builds them one after another to fix the previous tree's mistakes.

## 📺 Recommended Resources

* **StatQuest:** [Random Forests Part 1 - Building and Using](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
* **AssemblyAI:** [Implementing Random Forest from Scratch](https://www.youtube.com/watch?v=kFwe2ZZU7yw) (Great for the "Scratch" logic).
* **StatQuest:** [XGBoost Part 1: Regression](https://www.youtube.com/watch?v=OtD8wVaFm6E) (Even for classification, start here to get the logic).

## 🛠️ Your Mission

### Part A: The "From Scratch" Challenge

1. **The Tree:** Write a function that finds the best feature and best "split point" to separate survivors from non-survivors.
2. **The Forest:** Create a loop that builds `N` trees. For each tree:
   * Use `df.sample(replace=True)` to get a random subset of rows.
   * Limit the tree to a random subset of features (this is the "Random" in Random Forest).
3. **The Prediction:** For a new passenger, pass them through all your trees and take the "Mode" (the most common answer).

### Part B: The Library Speedrun (XGBoost)

1. **Install:** `pip install xgboost`.
2. **Implementation:** Use `XGBClassifier` on the same Titanic data.
3. **The Comparison:** Compare the accuracy of your "Scratch" model, the `sklearn` Random Forest, and `XGBoost`.
4. **Self
