
import cv2
import matplotlib.pyplot as plt
import numpy as np


class LemonDataset():
    def __init__(self):
        self.size = {'bad': 952, 'empty': 453, 'good': 1126}
        self.path= {'bad': 'lemon_dataset/bad_quality/bad_quality_', 'empty': 'lemon_dataset/empty_background/empty_background_', 'good': 'lemon_dataset/good_quality/good_quality_'}

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
            img_path=self.path[label]+str(idx[label])+'.jpg'
            img=cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i].imshow(img)
            ax[i].set_title(label)
            ax[i].axis('off')

    def __str__(self):
        self.show_samples()
        salida=str(self.size)
        
        return salida




