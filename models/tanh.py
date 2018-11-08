from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Concatenate, Dropout, Reshape, BatchNormalization
from keras.models import Model


class TanhModel:
    cell_line_size = 28087
    drug_size = 3072

    def __init__(self):
        self._model = self._generate_model()

    def _output_cell_line(self, input_cell_line):
        batch_norm_cell_line = BatchNormalization()(input_cell_line)
        conv_cell_line_1 = Conv1D(filters=50, kernel_size=700, strides=5, activation='tanh')(batch_norm_cell_line)
        maxpool_cell_line_1 = MaxPooling1D(pool_size=5)(conv_cell_line_1)
        conv_cell_line_2 = Conv1D(filters=30, kernel_size=5, strides=2, activation='relu')(maxpool_cell_line_1)
        maxpool_cell_line_2 = MaxPooling1D(pool_size=10)(conv_cell_line_2)
        flatten_cell_line = Flatten()(maxpool_cell_line_2)
        dense_cell_line = Dense(100, activation='relu')(flatten_cell_line)
        dropout_cell_line = Dropout(0.1)(dense_cell_line)
        return dropout_cell_line

    def _output_drug(self, input_drug):
        batch_norm_drug = BatchNormalization()(input_drug)
        conv_drug_1 = Conv1D(filters=50, kernel_size=200, strides=3, activation='tanh')(batch_norm_drug)
        maxpool_drug_1 = MaxPooling1D(pool_size=5)(conv_drug_1)
        conv_drug_2 = Conv1D(filters=30, kernel_size=50, strides=5, activation='relu')(maxpool_drug_1)
        maxpool_drug_2 = MaxPooling1D(pool_size=10)(conv_drug_2)
        flatten_drug = Flatten()(maxpool_drug_2)
        dense_drug = Dense(100, activation='relu')(flatten_drug)
        dropout_drug = Dropout(0.1)(dense_drug)
        return dropout_drug

    def _concatenate(self, outputs_cell_line, outputs_drug):
        concatenate = Concatenate()([outputs_cell_line, outputs_drug])
        dense_1 = Dense(300, activation='tanh')(concatenate)
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
        dense_2 = Dense(40, activation='relu')(flatten)
        dropout_3 = Dropout(0.2)(dense_2)
        dense_3 = Dense(10, activation='relu')(dropout_3)
        dropout_4 = Dropout(0.2)(dense_3)
        output = Dense(1, activation='tanh')(dropout_4)
        return output

    def _generate_model(self):
        input_cell_line = Input(shape=(self.cell_line_size, 1,), name='cell_line_input')
        output_cell_line = self._output_cell_line(input_cell_line)

        input_drug = Input(shape=(self.drug_size, 1,), name='drug_input')
        output_drug = self._output_drug(input_drug)

        output = self._concatenate(output_cell_line, output_drug)
        model = Model(inputs=[input_cell_line, input_drug], outputs=output)
        return model

    def __call__(self, *args, **kwargs):
        return self._model
