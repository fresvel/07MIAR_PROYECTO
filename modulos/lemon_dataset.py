import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LemonDataset():
    def __init__(self):
        self.dir_path= {'bad': 'lemon_dataset/bad_quality', 'empty': 'lemon_dataset/empty_background', 'good': 'lemon_dataset/good_quality'}
        self.img_path={label: self.dir_path[label]+'/'+ self.dir_path[label].split('/')[-1]+'_' for label in self.dir_path}
        self.size={}



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
        




    def __str__(self):
        self.show_samples()
        salida=str(self.size)
        return salida




