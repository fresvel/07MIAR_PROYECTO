import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modulos.lemon_dataset import LemonDataset

class LemonGenLoader(LemonDataset):
    def __init__(self, img_size=(224,224), batch_size=32, mode='scratch'):
        super().__init__(mode, "gen")
        self.img_size = img_size
        self.batch_size = batch_size
        self._create_splits()

        # Rango de aumentos
        #self.rotation_range = 25
        #self.zoom_range = (0.8, 1.2)
        #self.brightness_range = (0.8, 1.2)


    def get_generators(self):
        
        CLASSES = ["bad", "empty", "good"]
        base_datagen = ImageDataGenerator(rescale=1/255)
        
        aug_datagen = ImageDataGenerator(
            rescale=1/255,
            rotation_range=self.rotation_range,
            zoom_range=self.zoom_range,
            brightness_range=self.brightness_range,
            horizontal_flip=True
        )

        df_empty = self._train_df[self._train_df["class"] == "empty"]
        df_others = self._train_df[self._train_df["class"] != "empty"]

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

        # Combina ambos generadores
        def combined_gen():
            while True:
                X1, y1 = next(gen_empty)
                X2, y2 = next(gen_others)

                X = np.concatenate((X1, X2))
                y = np.concatenate((y1, y2))

                # Mezcla los lotes para evitar patrones de orden
                idx = np.arange(X.shape[0])
                np.random.shuffle(idx)
                yield X[idx], y[idx]

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

        # Calcula steps_per_epoch con equilibrio aproximado
        self.steps_per_epoch = max(len(gen_empty), len(gen_others))
        return combined_gen(), val_gen, test_gen

