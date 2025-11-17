import tensorflow as tf
from modulos.lemon_dataset import LemonDataset

class LemonTFLoader(LemonDataset):
    def __init__(self, img_size=(224,224), batch_size=32, mode='scratch'):
        LemonDataset.__init__(self, mode=mode, loader="tf")
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
        image = tf.image.random_contrast(image, self.contrast_range[0], self.contrast_range[1])
        image = tf.image.random_saturation(image, 0.6, 1.4)
        image = tf.image.random_hue(image, 0.08)
        crop_scale = tf.random.uniform([], 0.75, 0.95)
        image = tf.image.central_crop(image, crop_scale)
        image = tf.image.resize(image, self.img_size)
        return image, label


    def _conditional_augment(self, image, label):
        """
        Aplica augmentación solo si la clase es 'empty' (índice 1).
        """
        def augment():
            return image, label
            #return self._augment(image, label)

        def no_augment():
            return image, label
            #return self._augment(image, label)
        return tf.cond(tf.equal(tf.argmax(label), self.class_to_index['empty']), augment, no_augment)

    def get_old_datasets(self):
        """Crea y devuelve los datasets de entrenamiento, validación y prueba."""
        train_ds = tf.data.Dataset.from_tensor_slices(self.splits["train"])
        val_ds   = tf.data.Dataset.from_tensor_slices(self.splits["val"])
        test_ds  = tf.data.Dataset.from_tensor_slices(self.splits["test"])

        train_ds = train_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds   = val_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds  = test_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.map(self._conditional_augment, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.shuffle(512).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds   = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds  = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        return train_ds, val_ds, test_ds


    def get_datasets(self):
        """Crea datasets balanceados por batch usando sample_from_datasets."""
        # ---------------------------
        # 1. Crear datasets base
        # ---------------------------
        train_raw = tf.data.Dataset.from_tensor_slices(self.splits["train"])
        val_ds    = tf.data.Dataset.from_tensor_slices(self.splits["val"])
        test_ds   = tf.data.Dataset.from_tensor_slices(self.splits["test"])

        # procesamiento simple (sin augment todavía)
        train_raw = train_raw.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds    = val_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds   = test_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)

        # ---------------------------
        # 2. Subdividir por clase
        # ---------------------------
        ds_bad   = train_raw.filter(lambda x, y: tf.equal(tf.argmax(y), 0))
        ds_empty = train_raw.filter(lambda x, y: tf.equal(tf.argmax(y), 1))
        ds_good  = train_raw.filter(lambda x, y: tf.equal(tf.argmax(y), 2))

        # ---------------------------
        # 3. Mezclar con pesos balanceados
        #    (puedes subir empty ligeramente)
        # ---------------------------
        balanced_train = tf.data.Dataset.sample_from_datasets(
            [ds_bad, ds_empty, ds_good],
            weights=[0.33, 0.34, 0.33]   # puedes probar [0.25, 0.50, 0.25]
        )

        # ---------------------------
        # 4. Aplicar augment
        # ---------------------------
        balanced_train = balanced_train.map(
            self._conditional_augment,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # ---------------------------
        # 5. Batch + Prefetch
        # ---------------------------
        train_ds = (
            balanced_train
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds  = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # calcular steps_per_epoch
        self.steps_per_epoch = len(self.splits["train"]) // self.batch_size
        #self.steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()

        return train_ds, val_ds, test_ds


