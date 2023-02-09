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

    def create_dataframe(self, iteration, costs, w, b, djw, djb):
        data = []
        for i in range(len(iteration)):
            row = [iteration[i], costs[i], djb[i]]
            row = np.concatenate([row[:2], w[i], [b[i]], djw[i], row[-1:]])
            data.append(row)

        columns = ["iteration", "cost"]

        for i in range(w.shape[1]):
            columns.append(f'w{i}')

        columns.append('b')

        for i in range(djw.shape[1]):
            columns.append(f'dj_dw{i}')

        columns.append('dj_db')
        df = pd.DataFrame(data, columns=columns)
        return df

    def run(self, w, b, iter):
        m = self.X.shape[0]
        n = self.X.shape[1]
        costs = np.zeros(iter)
        dj_dw = np.zeros((iter, n))
        dj_db = np.zeros(iter)
        wi = np.zeros((iter, n))
        bi = np.zeros(iter)

        i = 0
        for i in range(iter):
            wi[i], bi[i], dj_dw[i], dj_db[i] = self.update_weights(
                wi[i-1], bi[i-1])
            costs[i] = self.cost(wi[i], bi[i])

        df = self.create_dataframe(
            np.arange(iter), costs, wi, bi, dj_dw, dj_db)
        return wi[-1], bi[-1], costs, df


class GradientDescentUnivariateLR():
    def __init__(self, X, y, lr=0.015):
        self.lr = lr
        self.X = X
        self.y = y
        self.m = X.shape[0]

    # Model representation
    def f(self, x, w, b):
        return w * x + b

    # Compute output prediction for each input sample, given some choice for w & b

    def compute_model_output(self, w, b):
        f_wb = np.zeros(self.m)

        for i in range(self.m):
            f_wb[i] = self.f(self.X[i], w, b)

        return f_wb

    def compute_cost(self, w, b):
        cost = self.compute_model_output(w, b) - self.y
        cost_sum = np.sum(cost**2)
        return (1 / (2 * self.m)) * cost_sum

    def cost_derivative_w(self, w, b):
        cost = self.compute_model_output(w, b) - self.y
        dj_dw = 0
        for i in range(self.X.shape[0]):
            dj_dw += cost[i] * self.X[i]
        dj_dw = dj_dw * (1 / self.m)
        return dj_dw

    def cost_derivative_b(self, w, b):
        cost = self.compute_model_output(w, b) - self.y
        dj_db = np.sum(cost)
        dj_db = dj_db * (1 / self.m)
        return dj_db

    def update_weights(self, w, b):
        tmp_w = w - self.lr * self.cost_derivative_w(w, b)
        tmp_b = b - self.lr * self.cost_derivative_b(w, b)
        w = tmp_w
        b = tmp_b

        return w, b

    def run(self, w, b, iter):
        cost = self.compute_cost(w, b)

        for i in range(iter):
            print(f'Current cost = {cost}')
            w, b = self.update_weights(w, b)
            cost = self.compute_cost(w, b)

        print(f'\nFinal cost = {cost}')
        return w, b
