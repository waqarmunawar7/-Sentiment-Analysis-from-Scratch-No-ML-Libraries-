# 🧠 Sentiment Analysis from Scratch (No ML Libraries)

This project implements a simple sentiment analysis classifier using **logistic regression**, trained using **manual gradient descent**, **without** relying on any machine learning libraries like scikit-learn or TensorFlow. It demonstrates end-to-end model development from raw text to evaluation.

---

## 📂 Project Structure

```
.
├── tweets.txt        # Your labeled tweet dataset
├── sentiment_analysis.py  
└── README.md                
```

---

## ✅ Features

- 📦 **Logistic Regression** from scratch  
- 📉 **Manual Gradient Descent** for weight updates  
- 🧹 **Text Preprocessing & Word Frequency Vectorization**  
- 📊 **Loss vs Epochs** graph  
- 🔁 **Parameter Convergence** plots  
- 📋 **Evaluation with Confusion Matrix & Custom Metrics**

---

## 📁 Input Format

The dataset file `tweets.txt` should contain tweets in the following format:

```
I love this product || Positive  
This is the worst thing ever || Negative  
```

Each line contains a tweet and its sentiment label (`Positive` or `Negative`) separated by `||`.

---

## 🧮 Time Complexity

| Component         | Complexity     | Description                                      |
|------------------|----------------|--------------------------------------------------|
| Vocabulary Build | O(N × L)       | N = # of tweets, L = avg. words per tweet       |
| Vectorization    | O(N × V)       | V = vocabulary size                             |
| Training Loop    | O(E × N × V)   | E = # of epochs (includes gradient computation) |
| Evaluation       | O(N × V)       | Same as vectorization for test set              |

---

## 🧠 Model Overview

We use a **logistic regression** model where:

```math
sigmoid(z) = 1 / (1 + exp(-z))
z = bias + w1 * pos_freq + w2 * neg_freq
```

**Gradient Descent Weight Update:**

```python
error = predicted - actual
w1 -= learning_rate * error * pos_freq
w2 -= learning_rate * error * neg_freq
bias -= learning_rate * error
```

---

## 📈 Output Graphs

- 📉 **Loss vs Epochs:** Shows how training error decreases over time  
- 📍 **Parameter Convergence:** Plots `w1`, `w2`, and `bias` vs loss with circle markers for better interpretability

---

## 📊 Evaluation Metrics

Evaluation is done using a **custom implementation**, without any external libraries:

- ✅ Accuracy
- 🔁 Precision
- 🎯 Recall
- 🧮 F1 Score
- 🧮 Confusion Matrix

---

## 🛠️ Requirements

- Python 3.6+
- `matplotlib` (for plotting)

```bash
pip install matplotlib
```

---

## 🚀 How to Run

```bash
python sentiment_analysis.py
```

---

## 🧪 Example Output

```bash
Weights: w1=0.45, w2=-0.27, bias=0.62
Confusion Matrix:
[[7, 2],
 [1, 10]]
Accuracy: 0.85
Precision: 0.83
Recall: 0.91
F1 Score: 0.87
```

---

## 🙌 Credits

Created by Waqar  
Inspired by hands-on ML principles and low-level learning

---
