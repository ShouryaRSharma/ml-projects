# Module 1: Linear Regression & Gradient Descent

**The Project:** California Housing Price Prediction
**The Goal:** Predict house values by manually coding the optimization process that finds the best-fit line.

## 🎯 Learning Objectives

* **The Hypothesis:** $y = wx + b$. Understand how changing $w$ (slope) and $b$ (intercept) moves the line.
* **Cost Function:** Manually calculate **Mean Squared Error (MSE)**.
* **The Derivative:** Understand that the gradient tells you which direction to move weights to reduce error.
* **Learning Rate ($\alpha$):** Discover how step size affects convergence.

## 📺 Recommended Resources

* **StatQuest:** [Linear Regression, Clearly Explained](https://www.youtube.com/watch?v=PaFPbb66DxQ)
* **StatQuest:** [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8)
* **Aladdin Persson:** [Linear Regression from Scratch](https://www.youtube.com/watch?v=pCCUnoes1Po)

## 🛠️ Your Mission

1. **From Scratch:** Initialize random $w$ and $b$. Use a loop to calculate the gradient of the MSE loss and update $w$ and $b$ manually using NumPy.
2. **Library:** Use `sklearn.linear_model.LinearRegression`.
3. **Comparison:** Compare your final weights vs. Scikit-Learn’s weights. Plot the "Loss Curve" to see the error decrease over iterations.
