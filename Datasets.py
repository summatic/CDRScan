import numpy as np


class Datasets:
    batch_size = 256
    cell_line_size = 28087
    drug_size = 3072

    def __init__(self, prefix, mode='test', k=5):
        self.prefix = prefix
        self.mode = mode
        self.train_X, self.train_y, self.test_X, self.test_y = self._load_dataset()
        self.k = k

    def _load_dataset(self):
        print('start loading dataset')
        if self.mode != 'test':
            train_X = np.load(self.prefix + 'CDRscan_data/train_X.npy')[::-1]
            train_y = np.load(self.prefix + 'CDRscan_data/train_y.npy')[::-1]
            test_X = np.load(self.prefix + 'CDRscan_data/test_X.npy')
            test_y = np.load(self.prefix + 'CDRscan_data/test_y.npy')
        else:
            train_X = np.load(self.prefix + 'CDRscan_data/train_X_small.npy')
            train_y = np.load(self.prefix + 'CDRscan_data/train_y_small.npy')
            test_X = np.load(self.prefix + 'CDRscan_data/test_X_small.npy')
            test_y = np.load(self.prefix + 'CDRscan_data/test_y_small.npy')

        train_X = np.expand_dims(train_X, axis=2)
        test_X = np.expand_dims(test_X, axis=2)

        print('end loading dataset')
        return train_X, train_y, test_X, test_y

    def _data_generator(self, X, y, split):
        for i in range(0, len(X), self.batch_size):
            ic50 = y[i:i+self.batch_size]

            if split:
                cell_line = X[i:i+self.batch_size, :self.cell_line_size, :]
                drug = X[i:i+self.batch_size, self.cell_line_size:, :]
                yield [cell_line, drug], ic50
            else:
                yield X[i:i+self.batch_size], ic50

    def _data_returner(self, X, y, split):
        if split:
            X_cell_line = X[:, :self.cell_line_size, :]
            X_drug = X[:, self.cell_line_size:, :]
            return [X_cell_line, X_drug], y
        else:
            return X, y

    def train_set(self, split=True):
        return self._data_generator(self.train_X, self.train_y, split)

    def test_set(self, split=True):
        return self._data_returner(self.test_X, self.test_y, split)

    def kfold(self, idx, split):
        maxlen = len(self.train_X) - len(self.train_X) % 5

        train_X = np.concatenate([fold for n, fold in enumerate(np.split(self.train_X[:maxlen], self.k)) if n != idx])
        train_y = np.concatenate([fold for n, fold in enumerate(np.split(self.train_y[:maxlen], self.k)) if n != idx])

        val_X = np.split(self.train_X[:maxlen], self.k)[idx]
        val_y = np.split(self.train_y[:maxlen], self.k)[idx]
        steps_per_epoch = len(train_y) // self.batch_size

        return self._data_generator(train_X, train_y, split), self._data_returner(val_X, val_y, split), steps_per_epoch
