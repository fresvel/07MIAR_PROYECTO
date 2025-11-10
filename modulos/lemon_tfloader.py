import tensorflow as tf
from modulos.lemon_dataset import LemonDataset

class LemonTFLoader(LemonDataset):
    def __init__(self, img_size=(224,224), batch_size=32, mode='scratch'):
        super().__init__(mode, "tf")
        self.img_size = img_size
        self.batch_size = batch_size
        self._create_splits()
        self.class_to_index = {"bad": 0, "empty": 1, "good": 2}



    def _process_path(self, file_path, label):
        #label = tf.cast(label, tf.string)
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        label = tf.one_hot(label, depth=3)
        return image, label

    def _augment(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, self.max_delta)
        image = tf.image.random_contrast(image, self.contrast_range[0],self.contrast_range[1])
        zoom_ratio = tf.random.uniform([], self.zoom_ratio[0], self.zoom_ratio[1])
        h, w, _ = image.shape
        image = tf.image.central_crop(image, zoom_ratio)
        image = tf.image.resize(image, self.img_size)
        return image, label

    def get_datasets(self):
        train_ds = tf.data.Dataset.from_tensor_slices(self.splits["train"])
        val_ds   = tf.data.Dataset.from_tensor_slices(self.splits["val"])
        test_ds  = tf.data.Dataset.from_tensor_slices(self.splits["test"])

        train_ds = train_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(512).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = val_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = test_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds
