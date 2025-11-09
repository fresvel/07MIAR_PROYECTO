import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd

from modulos.lemon_dataset import LemonDataset

class LemonGenLoader(LemonDataset):
    def __init__(self, img_size=(224,224), batch_size=32):
        super().__init__()
        self.img_size = img_size
        self.batch_size = batch_size

        # Creamos una lista global (img_path, label_string)
        self._collect_dataset()

    def _collect_dataset(self):
        images = []
        labels = []

        for label in self.dir_path:
            class_dir = self.dir_path[label]
            for img_name in os.listdir(class_dir):
                images.append(os.path.join(class_dir, img_name))
                labels.append(label)

        dataset = {'image':images, 'label':labels}
        self.dataframe=pd.DataFrame(dataset)
        self.images=self.dataframe['image'].values
        self.labels=self.dataframe['label'].values

    def _create_splits(self, test_size=0.15, val_size=0.15, seed=42):
        """
        Divide el dataset en tres subconjuntos: train, validation y test,
        manteniendo la proporción original de clases (estratificación).

        Parámetros
        ----------
        test_size : float, opcional (default=0.15)
            Proporción del conjunto total que se asignará al conjunto de prueba (test).
        
        val_size : float, opcional (default=0.15)
            Proporción del conjunto total que se asignará al conjunto de validación (validation).
        
        seed : int, opcional (default=42)
            Valor para la semilla aleatoria que garantiza la reproducibilidad de la división.

        Descripción
        -----------
        - Primero se separa el conjunto de entrenamiento (train), dejando un conjunto temporal
        que contiene los datos destinados a validación y prueba.
        - Luego, el conjunto temporal se divide nuevamente para obtener val y test,
        usando una proporción relativa entre ambos.
        - Se usa estratificación en ambas divisiones para mantener el balance entre clases.
        
        Resultado
        ---------
        Los subconjuntos se guardan en el atributo `self.splits` como un diccionario con la forma:

            self.splits = {
                "train": (X_train, y_train),
                "val":   (X_val,  y_val),
                "test":  (X_test, y_test)
            }
        """
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.images, self.labels, 
            train_size=(1-test_size-val_size),
            stratify=self.labels, random_state=seed
        )

        relative_val = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=relative_val,
            stratify=y_temp, random_state=seed
        )

        self.splits = {
            "train": (X_train, y_train),
            "val":   (X_val, y_val),
            "test":  (X_test, y_test)
        }


    def __str__(self):
        display(self.dataframe.sample(5))
        print(f"images: {type(self.images)}")
        print(f"labels: {type(self.labels)}")
        return ''
        
    


