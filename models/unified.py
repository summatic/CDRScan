from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Concatenate, Dropout, Reshape
from keras.models import Model


class UnifiedModel:
    cell_line_size = 28087
    drug_size = 3072

    def __init__(self):
        self._model = self._generate_model()

    def _output_unified(self, input_unified):
        conv_cell_line_1 = Conv1D(filters=50, kernel_size=700, strides=5, activation='tanh')(input_unified)
        maxpool_cell_line_1 = MaxPooling1D(pool_size=5)(conv_cell_line_1)
        conv_cell_line_2 = Conv1D(filters=30, kernel_size=5, strides=2, activation='relu')(maxpool_cell_line_1)
        maxpool_cell_line_2 = MaxPooling1D(pool_size=10)(conv_cell_line_2)
        flatten_cell_line = Flatten()(maxpool_cell_line_2)
        dense_cell_line = Dense(100, activation='relu')(flatten_cell_line)
        dropout_cell_line = Dropout(0.1)(dense_cell_line)
        return dropout_cell_line

    def _concatenate_unified(self, output_unified):
        dense_1 = Dense(300, activation='tanh')(output_unified)
        dropout_1 = Dropout(0.1)(dense_1)
        reshape_1 = Reshape((300, 1))(dropout_1)
        conv_1 = Conv1D(filters=30, kernel_size=150, strides=1, activation='relu')(reshape_1)
        maxpool_1 = MaxPooling1D(pool_size=2)(conv_1)
        conv_2 = Conv1D(filters=10, kernel_size=5, strides=1, activation='relu')(maxpool_1)
        maxpool_2 = MaxPooling1D(pool_size=3)(conv_2)
        conv_3 = Conv1D(filters=5, kernel_size=5, strides=1, activation='relu')(maxpool_2)
        maxpool_3 = MaxPooling1D(pool_size=3)(conv_3)
        dropout_2 = Dropout(0.1)(maxpool_3)
        flatten = Flatten()(dropout_2)
        dropout_3 = Dropout(0.2)(flatten)
        output = Dense(1, activation='linear')(dropout_3)
        return output

    def _generate_model(self):
        input_unified = Input(shape=(self.cell_line_size+self.drug_size, 1,), name='unified_input')
        _output_unified = self._output_unified(input_unified)

        output = self._concatenate_unified(_output_unified)
        model = Model(inputs=input_unified, outputs=output)
        return model

    def __call__(self, *args, **kwargs):
        return self._model
