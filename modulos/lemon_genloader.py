import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from modulos.lemon_dataset import LemonDataset

class LemonGenLoader(LemonDataset):
    def __init__(self, img_size=(224,224), batch_size=32, mode='scratch'):
        super().__init__(mode, "gen")
        self.img_size = img_size
        self.batch_size = batch_size
        

        # Creamos una lista global (img_path, label_string)
        self._create_splits()
      

    def get_generators(self):       
        train_datagen = ImageDataGenerator(
                rescale=1/255,
                rotation_range = self.rotation_range,
                zoom_range = self.zoom_range,
                brightness_range = self.brightness_range,
                horizontal_flip = True
            )
    

        test_val_datagen = ImageDataGenerator(rescale=1/255)

        train_gen = train_datagen.flow_from_dataframe(
            dataframe=self._train_df,
            x_col="filename",
            y_col="class",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True
        )
        
        test_gen = test_val_datagen.flow_from_dataframe(
            dataframe=self._test_df,
            x_col="filename",
            y_col="class",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False
        )

        val_gen = test_val_datagen.flow_from_dataframe(
            dataframe=self._val_df,
            x_col="filename",
            y_col="class",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False
        )

        return train_gen, val_gen, test_gen


    def __str__(self):
        #display(self.dataframe.sample(5))
        print(f"images: {type(self.images)}")
        print(f"labels: {type(self.labels)}")
        return ''
        
    


