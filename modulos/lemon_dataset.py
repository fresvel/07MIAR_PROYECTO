import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class LemonDataset():
    def __init__(self):
        self.dir_path= {'bad': 'lemon_dataset/bad_quality', 'empty': 'lemon_dataset/empty_background', 'good': 'lemon_dataset/good_quality'}
        self.img_path={label: self.dir_path[label]+'/'+ self.dir_path[label].split('/')[-1]+'_' for label in self.dir_path}
        self.size={}
        self._collect_dataset()
        


    def class_counter(self):
        for label in self.dir_path:
            self.size[label]=len(os.listdir(self.dir_path[label]))
        
        dic_size={label: [self.size[label]] for label in self.size}
        df_size=pd.DataFrame(dic_size).melt(var_name="Clase", value_name="Cantidad")
        display(df_size)


    def show_grid_per_class(self, n=9):
        for i in range(n):
            self.show_samples()    
    
    def show_samples(self):
        """
        Muestra un imágen aleatoria de cada clase del conjunto de datos.
    
        Esta función selecciona de manera aleatoria una imagen de cada una de las clases 
        disponibles en el conjunto de datos y las representa en una misma figura, lo que 
        permite realizar una inspección visual inicial del dataset.
    
        No retorna valores; simplemente despliega la figura resultante.
        """
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        idx = {label:np.random.randint(0, self.size[label]) for label in self.size}

        for i, label in enumerate(idx):
            img_path=self.img_path[label]+str(idx[label])+'.jpg'
            img=cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i].imshow(img)
            ax[i].set_title(label)
            ax[i].axis('off')


    def check_image_shapes(self):
        """
        Inspecciona las dimensiones de las imágenes para determinar si el dataset
        presenta tamaños uniformes o variables.

        Retorna:
            shapes_count (dict): Diccionario con las resoluciones encontradas y sus frecuencias.
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

        # Convertimos a DataFrame para mostrar bien
        df_shapes = pd.DataFrame([
            {"Dimensiones (H,W,C)": k, "Cantidad": v} 
            for k, v in shapes_count.items()
        ])

        display(df_shapes.sort_values("Cantidad", ascending=False))

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

        self._train_df = pd.DataFrame({"filename": X_train, "class": y_train})
        self._val_df   = pd.DataFrame({"filename": X_val,   "class": y_val})
        self._test_df  = pd.DataFrame({"filename": X_test,  "class": y_test})

        self.splits = {
            "train": (X_train, y_train),
            "val":   (X_val, y_val),
            "test":  (X_test, y_test)
        }


    def __str__(self):
        self.show_samples()
        salida=str(self.size)
        return salida




