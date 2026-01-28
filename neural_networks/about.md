# Module 4: Multi-Layer Perceptrons from Scratch

**The Project:** MNIST Digit Recognition
**The Goal:** Build a "vanilla" neural network using only `torch.nn` to identify digits 0-9.

## 🎯 Learning Objectives

* **`nn.Module`:** Understand how to structure a PyTorch class with an `__init__` (architecture) and a `forward` (execution) method.
* **Tensors & Shapes:** Learn why a 28x28 image must be "flattened" into a 784-dimension vector.
* **Loss & Optimization:** Differentiate between the loss function (`nn.CrossEntropyLoss`) and the optimizer (`optim.SGD` or `optim.Adam`).
* **The Training Loop:** Manually write the `zero_grad()`, `backward()`, and `step()` sequence.

## 📺 Recommended Resources

* **3Blue1Brown:** [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
* **StatQuest:** [Neural Networks in PyTorch](https://www.youtube.com/watch?v=MozHIoH6EKA)

## 🛠️ Your Mission

1. **Model:** Create a class inheriting from `nn.Module`.
2. **Layers:** Use `nn.Linear` for your layers and `nn.ReLU` for your activations.
3. **Data:** Use `torchvision.datasets.MNIST` and `DataLoader` to feed images in batches.
4. **Challenge:** Track your "Loss" over time using a simple list and plot it. If the line doesn't go down, your learning rate is likely wrong!
