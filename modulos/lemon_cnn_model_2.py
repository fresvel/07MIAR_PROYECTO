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
        x = SeparableConv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = SeparableConv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPooling2D()(x)
        x = Dropout(dropout_rate)(x)
        return x

    def build(self):
        inputs = Input(shape=self.input_shape)

        # *** NUEVO BLOQUE INICIAL ***
        x = tf.keras.layers.Conv2D(32, 3, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)

        x = self._sep_block(x, 64, 0.10)
        x = self._sep_block(x, 128, 0.15)
        x = self._sep_block(x, 256, 0.20)

        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-5))(x)
        x = Dropout(0.3)(x)

        outputs = Dense(self.num_classes, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs, name="lemon_cnn_separable_v2")
