import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class GradientDescentMultipleLR():

    def __init__(self, X, y, lr=0.1):
        self.X = X
        self.y = y
        self.lr = lr

    def f(self, x, w, b):
        return np.dot(x, w) + b

    def compute_model_output(self, w, b):
        m = self.X.shape[0]
        outputs = np.zeros(m)

        for i in range(m):
            outputs[i] = self.f(self.X[i], w, b)

        return outputs

    def cost(self, w, b):
        m = self.X.shape[0]
        cost_sum = np.sum((self.compute_model_output(w, b) - self.y)**2)
        cost = cost_sum / (2*m)
        return cost

    def compute_gradient(self, w, b):
        m = self.X.shape[0]
        err = self.compute_model_output(w, b) - self.y
        dj_db = np.sum(err) / m
        dj_dwi = np.zeros(len(w))

        for i in range(len(w)):
            dj_dwi[i] = np.sum(err * self.X[:, i]) / m

        return dj_dwi, dj_db

    def update_weights(self, w, b):
        dj_dwi, dj_db = self.compute_gradient(w, b)
        w = w - self.lr * dj_dwi
        b = b - self.lr * dj_db
        return w, b, dj_dwi, dj_db

    def create_dataframe(self, iteration, costs, djw, djb):
        data = []
        for i in range(len(iteration)):
            row = [iteration[i], costs[i], djb[i]]
            row = np.concatenate([row[:2], djw[i], row[-1:]])
            data.append(row)

        columns = ["iteration", "cost"]
        for i in range(djw.shape[1]):
            columns.append(f'dj_dw{i}')
        columns.append('dj_db')

        df = pd.DataFrame(
            data, columns)
        return df

    def run(self, w, b, iter):
        m = self.X.shape[0]
        n = self.X.shape[1]
        costs = np.zeros(iter)
        dj_dw = np.zeros((m, n))
        dj_db = np.zeros(m)

        print(f'Initial Cost = {self.cost(w, b)}')

        for i in range(iter):
            w, b, dj_dw[i], dj_db[i] = self.update_weights(w, b)
            costs[i] = self.cost(w, b)

        print(self.create_dataframe(np.arange(iter), costs, dj_dw, dj_db))
        return w, b, costs
