import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from sklearn.model_selection import train_test_split
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

        dataset = {'image':self.images, 'label':self.labels}
        self.dataframe=pd.DataFrame(dataset)
        self.images=self.dataframe['image'].values
        self.labels=self.dataframe['label'].values

