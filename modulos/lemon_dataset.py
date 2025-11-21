"""Módulo con utilidades para cargar y explorar el dataset de limones.

`LemonDataset` proporciona funciones para recopilar rutas de imágenes,
inspeccionar el dataset, y generar splits estratificados para entrenamiento,
validación y prueba.

Notas:
- Las rutas por defecto apuntan a subdirectorios dentro de `lemon_dataset/`.
- Soporta dos modos de uso de loader: `'tf'` (para `tf.data`) y `'gen'`
  (para `ImageDataGenerator`), controlado por el parámetro `loader`.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class LemonDataset():
    """Clase base para manejar el dataset de limones.

    Esta clase se usa como punto de partida para distintos loaders
    (`LemonTFLoader`, `LemonGenLoader`) y centraliza:
    - la recolección de rutas de imagen y etiquetas
    - parámetros de augmentation por defecto
    - creación de splits estratificados

    Args:
        mode (str): 'scratch' o cualquier otro valor para ajustar
            hiperparámetros de augmentation.
        loader (str): 'tf' o 'gen' para indicar el tipo de loader
            que usará las rutas recopiladas.
    """

    def __init__(self, mode='scratch', loader='tf'):
        # Mapas de directorios por clase (se pueden ajustar si cambia la estructura)
        self.dir_path = {
            'bad': 'lemon_dataset/bad_quality',
            'empty': 'lemon_dataset/empty_background',
            'good': 'lemon_dataset/good_quality'
        }

        # Prefijo de nombre de fichero asumido por el dataset (ej. 'bad/bad_')
        self.img_path = {
            label: self.dir_path[label] + '/' +
            self.dir_path[label].split('/')[-1] + '_'
            for label in self.dir_path
        }

        self.size = {}
        self.loader = loader

        # Parámetros de augmentation por defecto, según modo
        if mode == 'scratch':
            self.rotation_range = 20
            self.zoom_range = (0.75, 1.0)
            self.brightness_range = (0.7, 1.3)
            self.max_delta = 0.15
            self.contrast_range = (0.8, 1.2)
            self.zoom_ratio = (0.85, 0.95)
        else:
            self.rotation_range = 10
            self.zoom_range = 0.07
            self.brightness_range = (0.9, 1.1)
            self.max_delta = 0.07
            self.contrast_range = (0.9, 1.1)
            self.zoom_ratio = (0.90, 0.98)

        # Recolectar rutas de imágenes y etiquetas en `self.dataframe`
        self._collect_dataset()

    def class_counter(self):
        """Cuenta imágenes por clase y muestra un DataFrame con los totales."""
        for label in self.dir_path:
            self.size[label] = len(os.listdir(self.dir_path[label]))

        dic_size = {label: [self.size[label]] for label in self.size}
        df_size = pd.DataFrame(dic_size).melt(
            var_name="Clase", value_name="Cantidad")
        display(df_size)

    def show_grid_per_class(self, n=9):
        """Muestra `n` veces una muestra aleatoria por clase (usa `show_samples`)."""
        for _ in range(n):
            self.show_samples()

    def show_samples(self):
        """Muestra una imagen aleatoria de cada clase en una figura.

        Esta función es útil para una inspección visual rápida del dataset.
        """
        g = np.random.default_rng()  # NOSONAR(S6709) # generador aleatorio
        _, ax = plt.subplots(1, 3, figsize=(18, 5))
        idx = {label: g.integers(
            0, self.size[label]) for label in self.size}

        for i, label in enumerate(idx):
            img_path = self.img_path[label] + str(idx[label]) + '.jpg'
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i].imshow(img)
            ax[i].set_title(label)
            ax[i].axis('off')

    def check_image_shapes(self):
        """Recorre las imágenes y cuenta las dimensiones encontradas.

        Retorna un DataFrame que muestra las resoluciones y la cantidad
        de imágenes con cada resolución.
        """
        shapes_count = {}  # (alto, ancho, canales) -> cantidad

        for label in self.dir_path:
            class_path = self.dir_path[label]

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Error en la imagen: {img_path}")
                    continue  # evitar errores si hay archivos corruptos

                h, w, c = img.shape
                key = (h, w, c)
                shapes_count[key] = shapes_count.get(key, 0) + 1

        df_shapes = pd.DataFrame([
            {"Dimensiones (H,W,C)": k, "Cantidad": v}
            for k, v in shapes_count.items()
        ])

        display(df_shapes.sort_values("Cantidad", ascending=False))

    def _collect_dataset(self):
        """Recopila rutas de imagen y etiquetas en `self.dataframe`.

        Crea además arrays `self.images` y `self.labels`. Si `loader=='tf'`
        añade la columna `label_id` con códigos enteros para cada categoría.
        """
        images = []
        labels = []

        for label in self.dir_path:
            class_dir = self.dir_path[label]
            for img_name in os.listdir(class_dir):
                images.append(os.path.join(class_dir, img_name))
                labels.append(label)

        dataset = {'image': images, 'label': labels}
        self.dataframe = pd.DataFrame(dataset)
        self.dataframe['label'] = self.dataframe['label'].astype('category')
        self.images = self.dataframe['image'].values
        self.labels = self.dataframe['label'].values

        if self.loader == 'tf':
            # Codificar etiquetas como enteros (útil para tf.data)
            self.dataframe['label_id'] = self.dataframe['label'].cat.codes
            self.labels = self.dataframe['label_id'].astype('int32').values
        elif self.loader == 'gen':
            # Para ImageDataGenerator se mantienen las etiquetas como strings
            self.labels = self.dataframe['label'].values
        else:
            # corregido: usar self.loader en lugar de self.cfg
            raise ValueError(
                f"loader '{self.loader}' no reconocido. Use 'gen' o 'tf'.")

    def _create_splits(self, test_size=0.15, val_size=0.15, seed=42):
        """Divide el dataset en train/val/test manteniendo estratificación.

        Devuelve los splits en `self._train_df`, `self._val_df`, `self._test_df`
        y guarda tuplas en `self.splits`.
        """
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.images, self.labels,
            train_size=(1 - test_size - val_size),
            stratify=self.labels, random_state=seed
        )

        relative_val = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=relative_val,
            stratify=y_temp, random_state=seed
        )

        self._train_df = pd.DataFrame({"filename": X_train, "class": y_train})
        self._val_df = pd.DataFrame({"filename": X_val, "class": y_val})
        self._test_df = pd.DataFrame({"filename": X_test, "class": y_test})

        self.splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test)
        }

    def __str__(self):
        """Mostrar una muestra y devolver información compacta"""
        self.show_samples()
        salida = str(self.size)
        return salida
