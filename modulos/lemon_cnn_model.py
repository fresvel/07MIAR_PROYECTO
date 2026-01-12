"""Modulo `lemon_cnn_model`.

Provee la clase `LemonCNNBuilder` para construir un modelo CNN
ligero basado en SeparableConv2D orientado a clasificación
de imágenes (por defecto, 3 clases).

Contenido:
- `LemonCNNBuilder`: builder que genera un `tf.keras.Model`.

El estilo de la red usa bloques de convoluciones separables,
normalización por batch, activaciones ReLU, pooling y dropout.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, SeparableConv2D, BatchNormalization, ReLU,
    MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.regularizers import l2


class LemonCNNBuilder:
    """Builder para un modelo CNN compacto usando convoluciones separables.

    Esta clase encapsula la construcción de una arquitectura que aplica
    bloques de `SeparableConv2D` + normalización + ReLU, seguidos de
    pooling y dropout. Proporciona un método `build` que devuelve
    una instancia de `tf.keras.Model` lista para compilar.
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        """Inicializa el builder.

        Args:
            input_shape (tuple): Dimensiones de la entrada, por defecto (224, 224, 3).
            num_classes (int): Número de clases de salida para la capa final.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _sep_block(self, x, filters, dropout_rate):
        """Bloque de convoluciones separables.

        Construye dos capas `SeparableConv2D` (cada una seguida de
        `BatchNormalization` y `ReLU`), seguidas de `MaxPooling2D` y
        `Dropout`.

        Args:
            x (tf.Tensor): Tensor de entrada al bloque.
            filters (int): Número de filtros para las convoluciones.
            dropout_rate (float): Tasa de dropout aplicada tras el pooling.

        Returns:
            tf.Tensor: Tensor de salida del bloque.
        """
        # Primera convolución separable + normalización y activación
        x = SeparableConv2D(filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Segunda convolución separable + normalización y activación
        x = SeparableConv2D(filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Pooling espacial y regularización por dropout
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(dropout_rate)(x)
        return x

    def build(self):
        """Construye y devuelve el modelo Keras.

        La arquitectura consiste en una sucesión de bloques separables
        con números crecientes de filtros, un GlobalAveragePooling y
        una cabeza densa ligera para clasificación.

        Returns:
            tf.keras.Model: Modelo compilable de Keras (sin compilar).
        """
        inputs = Input(shape=self.input_shape)

        # Pila de bloques separables con aumento de filtros
        x = self._sep_block(inputs, 32, 0.25)
        x = self._sep_block(x, 64, 0.30)
        x = self._sep_block(x, 128, 0.35)
        x = self._sep_block(x, 256, 0.40)

        # Capas finales: pooling global, densa con regularización y salida
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.4)(x)

        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs, name="lemon_cnn_separable")
        return model
