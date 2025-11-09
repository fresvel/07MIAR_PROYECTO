from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class LemonTFLoader(LemonDataset):
    def __init__(self, img_size=(224,224), batch_size=32):
        super().__init__()
        self.img_size = img_size
        self.batch_size = batch_size

        self._collect_dataset()

    def _collect_dataset(self):
        self.images = []
        self.labels = []

        for label in self.dir_path:
            class_dir = self.dir_path[label]
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def create_splits(self, test_size=0.15, val_size=0.15, seed=42):
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.images, self.labels, 
            test_size=(test_size+val_size), 
            stratify=self.labels, random_state=seed
        )

        relative_val = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=relative_val, 
            stratify=y_temp, random_state=seed
        )

        self.splits = {
            "train": (X_train, y_train),
            "val":   (X_val, y_val),
            "test":  (X_test, y_test)
        }

    def _process_path(self, file_path, label):
        label = tf.cast(label, tf.string)
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def _augment(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        return image, label

    def get_datasets(self):
        train_paths, train_labels = self.splits["train"]
        val_paths, val_labels     = self.splits["val"]
        test_paths, test_labels   = self.splits["test"]

        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        val_ds   = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        test_ds  = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

        train_ds = train_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(512).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = val_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = test_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds
