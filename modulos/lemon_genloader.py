"""Loader de datos basado en generadores Keras para el dataset de limones.

Esta implementación extiende `LemonDataset` y proporciona generadores
de entrenamiento, validación y test usando `ImageDataGenerator`.

Principales características:
- Separación del conjunto de entrenamiento en subconjuntos para aplicar
  aumentos sólo a la clase `empty` (ejemplo de balanceo/augmentation dirigida).
- `combined_gen` mezcla los batches de las dos fuentes para producir
  lotes combinados y mezclados.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modulos.lemon_dataset import LemonDataset


class LemonGenLoader(LemonDataset):
    """Generador de datos para entrenamiento con aumentos selectivos.

    Este loader crea tres generadores:
    - entrenamiento (generator combinado que junta imágenes aumentadas de
      la clase `empty` con las restantes sin aumento)
    - validación (sin aumento, solo rescale)
    - test (sin aumento, solo rescale)

    Args:
        img_size (tuple): Tamaño objetivo para las imágenes (alto, ancho).
        batch_size (int): Tamaño de lote para los generadores.
        mode (str): Modo heredado para `LemonDataset` (por defecto 'scratch').
    """

    def __init__(self, img_size=(224, 224), batch_size=32, mode='scratch'):
        super().__init__(mode, "gen")
        self.img_size = img_size
        self.batch_size = batch_size
        # Crea los splits y DataFrames internos (_train_df, _val_df, _test_df)
        self._create_splits()

        # Rango de aumentos (pueden descomentarse y ajustarse antes de usar)
        # self.rotation_range = 25
        # self.zoom_range = (0.8, 1.2)
        # self.brightness_range = (0.8, 1.2)

    def get_generators(self):
        """Construye y devuelve los generadores de entrenamiento/validación/test.

        Returns:
            tuple: `(train_generator, val_generator, test_generator)` donde
            `train_generator` es un generador Python (yielding batches), y
            `val_generator` / `test_generator` son objetos `DirectoryIterator`
            producidos por `ImageDataGenerator.flow_from_dataframe`.
        """

        # Clases explícitas y normalización básica (rescale)
        CLASSES = ["bad", "empty", "good"]
        base_datagen = ImageDataGenerator(rescale=1/255)

        # DataGenerator para aumentos: usa atributos del objeto (si se configuran)
        aug_datagen = ImageDataGenerator(
            rescale=1/255,
            rotation_range=getattr(self, 'rotation_range', 0),
            zoom_range=getattr(self, 'zoom_range', None),
            brightness_range=getattr(self, 'brightness_range', None),
            horizontal_flip=True
        )

        # Separa el DataFrame de entrenamiento para aplicar aumentos sólo a `empty`
        df_empty = self._train_df[self._train_df["class"] == "empty"]
        df_others = self._train_df[self._train_df["class"] != "empty"]

        # Generador con aumentos para la clase `empty`
        gen_empty = aug_datagen.flow_from_dataframe(
            dataframe=df_empty,
            x_col="filename",
            y_col="class",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            classes=CLASSES,
            shuffle=True
        )

        # Generador base (sin aumento) para el resto de clases
        gen_others = base_datagen.flow_from_dataframe(
            dataframe=df_others,
            x_col="filename",
            y_col="class",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            classes=CLASSES,
            shuffle=True
        )

        # Combina ambos generadores en un solo flujo mezclado.
        # Esto devuelve un generador Python que yield batches mezclados
        # compuestos por ejemplos aumentados (empty) y no aumentados.
        def combined_gen():
            """Generador infinito que combina y mezcla batches de dos fuentes."""
            while True:
                X1, y1 = next(gen_empty)
                X2, y2 = next(gen_others)

                # Concatenar y mezclar para evitar sesgos de orden
                X = np.concatenate((X1, X2))
                y = np.concatenate((y1, y2))

                idx = np.arange(X.shape[0])
                np.random.shuffle(idx)
                yield X[idx], y[idx]

        # Validación y test sin aumentos (solo rescale)
        val_gen = base_datagen.flow_from_dataframe(
            dataframe=self._val_df,
            x_col="filename",
            y_col="class",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            classes=CLASSES,
            shuffle=False
        )

        test_gen = base_datagen.flow_from_dataframe(
            dataframe=self._test_df,
            x_col="filename",
            y_col="class",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            classes=CLASSES,
            shuffle=False
        )

        # Estimación de steps_per_epoch: usar el mayor de ambos generadores
        # para cubrir el conjunto de datos combinando las fuentes.
        # Nota: `len(gen)` es el número de batches por epoch para cada generator.
        try:
            self.steps_per_epoch = max(len(gen_empty), len(gen_others))
        except Exception:
            # Si por algún motivo los iteradores no soportan __len__, dejar None
            self.steps_per_epoch = None

        return combined_gen(), val_gen, test_gen
