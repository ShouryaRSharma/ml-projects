# Module 5: Transformers & Attention (PyTorch)

**The Project:** Sequence Classification (Sentiment Analysis)
**The Goal:** Instead of using a pre-made model, use PyTorch's `nn.TransformerEncoder` to build the brain itself.

## 🎯 Learning Objectives

* **Self-Attention:** Understand the Query ($Q$), Key ($K$), and Value ($V$) relationship.
* **Positional Encoding:** Since Transformers process all words at once, learn how we "tell" the model which word came first.
* **`nn.TransformerEncoderLayer`:** Learn how to stack multiple attention layers together.
* **Word Embeddings:** Use `nn.Embedding` to turn vocabulary indices into dense vectors.

## 📺 Recommended Resources

* **StatQuest:** [Transformer Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=zxQyTK8quyY)
* **Aladdin Persson:** [Transformer from Scratch (PyTorch)](https://www.youtube.com/watch?v=U0s0f995w14) (Highly recommended for pure PyTorch fans).

## 🛠️ Your Mission

1. **Model:** Build a `TransformerClassifier` class.
2. **Components:** Inside your model, include an `nn.Embedding` layer, a `PositionalEncoding` function, and an `nn.TransformerEncoder`.
3. **The "Head":** Add a final `nn.Linear` layer that takes the Transformer's output and squashes it into a 1 (positive) or 0 (negative).
4. **Tokenizer:** You can use a simple `torchtext` tokenizer or even a basic `.split()` for this toy project to keep it 100% PyTorch-centric.
5. **Challenge:** Compare the training time of this model vs. the Random Forest from Module 3. (Spoiler: Transformers are "hungry" for data and compute!).
