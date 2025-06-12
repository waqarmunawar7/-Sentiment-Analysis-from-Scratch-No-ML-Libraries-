import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from math import exp, log
import numpy as np


class SentimentAnalysisModel:
    def __init__(self, dataset_path, learning_rate=0.1, epochs=1000):
        self.dataset_path = dataset_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w1 = 0
        self.w2 = 0
        self.bias = 1
        self.losses = []
        self.w1_history = []
        self.w2_history = []
        self.bias_history = []

    def load_dataset(self):
        with open(self.dataset_path) as f:
            raw_data = f.readlines()
        x, y = [], []
        for tweet in raw_data:
            text, label = tweet.split('||')
            x.append(text.strip())
            y.append(label.strip())
        return x, y, raw_data

    def train_test_split(self, x, y, test_ratio=0.2):
        data = list(zip(x, y))
        np.random.seed(42)
        np.random.shuffle(data)
        split_idx = int(len(data) * (1 - test_ratio))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        X_train, y_train = zip(*train_data)
        X_test, y_test = zip(*test_data)
        return list(X_train), list(X_test), list(y_train), list(y_test)

    def build_vocab_freq(self, raw_data):
        tweet_separate = defaultdict(str)
        vocabulary = set()
        for tweet in raw_data:
            text, label = tweet.split('||')
            label = label.strip()
            words = [w for w in text.strip().split() if w]
            tweet_separate[label] += ' ' + text
            vocabulary.update(words)
        label_word_counts = {label: Counter(tweet_separate[label].split()) for label in tweet_separate}
        return {
            word: (
                label_word_counts.get('Positive', Counter()).get(word, 0),
                label_word_counts.get('Negative', Counter()).get(word, 0)
            )
            for word in vocabulary if word
        }

    def vectorize(self, vocabulary_term_freq, labels, tweet_samples):
        for i, tweet in enumerate(tweet_samples):
            pos_sum = 0
            neg_sum = 0
            for word in tweet.strip().split():
                if word in vocabulary_term_freq:
                    pos_sum += vocabulary_term_freq[word][0]
                    neg_sum += vocabulary_term_freq[word][1]
            label_value = 1 if labels[i] == "Positive" else 0
            max_val = max(pos_sum, neg_sum, 1)
            yield [pos_sum / max_val, neg_sum / max_val, label_value]

    def compute_input(self, input_vector):
        x = self.bias + (self.w1 * input_vector[0]) + (self.w2 * input_vector[1])
        return 1 / (1 + exp(-x))

    def train(self, x_train, y_train, vocabulary_term_freq):
        for epoch in range(self.epochs):
            total_loss = 0
            for i, vec in enumerate(self.vectorize(vocabulary_term_freq, y_train, x_train)):
                y_true = vec[2]
                y_pred = self.compute_input(vec)
                y_pred = max(min(y_pred, 1 - 1e-15), 1e-15)

                loss = - (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
                total_loss += loss

                error = y_pred - y_true
                self.w1 -= self.learning_rate * error * vec[0]
                self.w2 -= self.learning_rate * error * vec[1]
                self.bias -= self.learning_rate * error

            self.losses.append(total_loss)
            self.w1_history.append(self.w1)
            self.w2_history.append(self.w2)
            self.bias_history.append(self.bias)

    def evaluate(self, x_data, y_data, vocabulary_term_freq):
        y_pred = []
        y_true = []
        for i, vec in enumerate(self.vectorize(vocabulary_term_freq, y_data, x_data)):
            pred = self.compute_input(vec)
            label = 1 if pred > 0.5 else 0
            y_pred.append(label)
            y_true.append(vec[2])

        tp = sum((p == 1 and t == 1) for p, t in zip(y_pred, y_true))
        tn = sum((p == 0 and t == 0) for p, t in zip(y_pred, y_true))
        fp = sum((p == 1 and t == 0) for p, t in zip(y_pred, y_true))
        fn = sum((p == 0 and t == 1) for p, t in zip(y_pred, y_true))

        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        return {
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    def plot_convergence(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].plot(range(1, self.epochs + 1), self.losses, marker='o')
        ax[0].set_title("Cost Function over Epochs")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Total Loss")

        ax[1].scatter(self.w1_history, self.losses, c='r', label='w1', marker='o')
        ax[1].scatter(self.w2_history, self.losses, c='g', label='w2', marker='o')
        ax[1].scatter(self.bias_history, self.losses, c='b', label='bias', marker='o')
        ax[1].set_title("Parameter Convergence")
        ax[1].set_xlabel("Parameter Value")
        ax[1].set_ylabel("Loss")
        ax[1].legend()

        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    model = SentimentAnalysisModel("tweets.txt")
    x, y, raw_data = model.load_dataset()
    x_train, x_test, y_train, y_test = model.train_test_split(x, y)
    vocab = model.build_vocab_freq(raw_data)

    model.train(x_train, y_train, vocab)
    evaluation = model.evaluate(x_train, y_train, vocab)

    print("Weights:", model.w1, model.w2, model.bias)
    print("Confusion Matrix:")
    for row in evaluation["confusion_matrix"]:
        print(row)
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"F1 Score: {evaluation['f1_score']:.4f}")

    model.plot_convergence()
