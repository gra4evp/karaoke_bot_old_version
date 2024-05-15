import csv
import numpy as np
import random


class SparseSVD:
    def __init__(self, ratings_matrix, num_factors=10, lr=0.01, epochs=10):
        self.ratings_matrix = ratings_matrix
        self.num_factors = num_factors
        self.lr = lr

        self.epochs = epochs

        self.indices = list(zip(*np.where(ratings_matrix != 0)))

        num_users, num_items = ratings_matrix.shape
        # Инициализация латентных представлений пользователей и предметов
        self.P = np.random.normal(scale=1. / num_factors, size=(num_users, num_factors))
        self.Q = np.random.normal(scale=1. / num_factors, size=(num_items, num_factors))

        # Вычисляем средние значения по пользователям и предметам
        self.user_means = np.mean(ratings_matrix, axis=1)
        self.item_means = np.mean(ratings_matrix, axis=0)

    def forward(self, u, i):
        # Определяем рейтинг, центрированный по пользователям и предметам
        centered_rating = self.ratings_matrix[u, i] - self.user_means[u] - self.item_means[i]
        error = centered_rating - self.predict(u, i)
        return error

    def backward(self, error, u, i):
        # Делаем шаг градиента
        self.P[u] += self.lr * error * self.Q[i]
        self.Q[i] += self.lr * error * self.P[u]

    def fit(self):
        for _ in range(self.epochs):
            # Перебираем элементы в случайном порядке
            random.shuffle(self.indices)
            for u, i in self.indices:
                error = self.forward(u, i)
                self.backward(error, u, i)

    def predict(self, user_id, item_id):
        if user_id >= self.P.shape[0] or item_id >= self.Q.shape[0]:
            return 0
        return np.dot(self.P[user_id], self.Q[item_id])


class LFM(SparseSVD):
    def __init__(self, ratings_matrix, num_factors=10, lr=0.01, epochs=10, reg_params=(0.01, 0.01)):
        super().__init__(ratings_matrix, num_factors=num_factors, lr=lr, epochs=epochs)
        self.reg_p, self.reg_q = reg_params

    def backward(self, error, u, i):
        # Добавляем шаг градиента с учетом регуляризации
        self.P[u] += self.lr * (error * self.Q[i] - self.reg_p * self.P[u])
        self.Q[i] += self.lr * (error * self.P[u] - self.reg_q * self.Q[i])


class NonNegativeLFM(LFM):
    def backward(self, error, u, i):
        # Делаем шаг против градиента
        super().backward(error, u, i)

        # Проекция градиента на неотрицательную область
        self.P[u] = np.maximum(0, self.P[u])
        self.Q[i] = np.maximum(0, self.Q[i])


def update_dicts_with_idxs(filename):
    global user_idx, url_idx, num_records
    with open(filename, encoding='utf-8') as file:
        for line in csv.DictReader(file):
            user_id, url = line['user_id'], line['url']

            if user_id not in user_id2idx:
                user_id2idx[user_id] = [user_idx, 1]  # idx and freq for user
                user_idx += 1
            else:
                user_id2idx[user_id][1] += 1

            if url not in url2idx:
                url2idx[url] = [url_idx, 1]  # idx and freq for url
                url_idx += 1
            else:
                url2idx[url][1] += 1

            num_records += 1


def fill_rating_matrix(filename):
    global R
    with open(filename, encoding='utf-8') as file:
        for line in csv.DictReader(file):
            user_idx, num_orders = user_id2idx[line['user_id']]
            url_idx, freq = url2idx[line['url']]

            R[user_idx, url_idx] += 1


def get_top_user_ratings(user_id, model, top_n=None):
    ratings = []
    for track_id in range(model.R.shape[1]):
        ratings.append((track_id, round(model.predict(user_id, track_id), 4)))

    ranked_ratings = sorted(ratings, key=lambda item: -item[1])
    if top_n is None:
        return ranked_ratings
    return ranked_ratings[:top_n]


id_url_all_csv = '/app/pgrachev/data/id_url_all.csv'
visitor_performance_csv = '/app/pgrachev/data/visitor_performance.csv'


user_idx, user_id2idx = 0, {}
url_idx, url2idx = 0, {}
num_records = 0

# В начале заполняем словари для доп данных чтобы они были первые по дате
# Потому что они были сделаны до БД
update_dicts_with_idxs(filename=id_url_all_csv)
update_dicts_with_idxs(filename=visitor_performance_csv)  # Обновляем словари уже с основными данными

urls = list(url2idx.keys())

R = np.zeros((len(user_id2idx), len(url2idx)))
fill_rating_matrix(filename=id_url_all_csv)
fill_rating_matrix(filename=visitor_performance_csv)


model = NonNegativeLFM(R, num_factors=20, lr=0.01, epochs=10)
model.fit()
