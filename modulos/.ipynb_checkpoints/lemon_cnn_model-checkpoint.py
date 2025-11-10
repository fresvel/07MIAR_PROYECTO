import tensorflow as tf
from tensorflow.keras.layers import (
    Input, SeparableConv2D, BatchNormalization, ReLU,
    MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.regularizers import l2


class LemonCNNBuilder:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _sep_block(self, x, filters, dropout_rate):
        # Separable Convolution Block × 2
        x = SeparableConv2D(filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = SeparableConv2D(filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(dropout_rate)(x)
        return x

    def build(self):
        inputs = Input(shape=self.input_shape)

        x = self._sep_block(inputs, 32, 0.25)
        x = self._sep_block(x, 64, 0.30)
        x = self._sep_block(x, 128, 0.35)
        x = self._sep_block(x, 256, 0.40)

        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.4)(x)

        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs, name="lemon_cnn_separable")
        return model
