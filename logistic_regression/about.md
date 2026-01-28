# Module 2: Logistic Regression (Classification)

**The Project:** Breast Cancer Diagnostic Tool
**The Goal:** Classify a tumor as "Malignant" or "Benign" by squashing linear outputs into probabilities.

## 🎯 Learning Objectives

* **The Sigmoid Function:** $1 / (1 + e^{-z})$. Learn how to map $(-\infty, \infty)$ to $(0, 1)$.
* **Log Loss (Cross-Entropy):** Why MSE is bad for classification and why we use Log Loss instead.
* **Decision Boundary:** Understanding that the model picks a threshold (usually 0.5) to decide the class.

## 📺 Recommended Resources

* **StatQuest:** [Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
* **StatQuest:** [Odds and Log(Odds)](https://www.youtube.com/watch?v=ARfXDSkQf1Y)

## 🛠️ Your Mission

1. **From Scratch:** Implement the Sigmoid function. Update your Gradient Descent loop from Module 1 to use the Logistic Loss derivative.
2. **Library:** Use `sklearn.linear_model.LogisticRegression`.
3. **Evaluation:** Build a **Confusion Matrix**. Calculate Precision and Recall. Why is a "False Negative" (missing a cancer case) worse than a "False Positive" here?
