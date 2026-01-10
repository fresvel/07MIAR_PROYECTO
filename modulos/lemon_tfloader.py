"""Utilities para cargar el dataset usando tf.data.

`LemonTFLoader` extiende `LemonDataset` y expone interfaces para crear
`tf.data.Dataset` preparados para entrenamiento, validación y test.

Características principales:
- Procesado eficiente con `tf.data` y `AUTOTUNE`.
- Augmentation condicional (solo para la clase `empty`) para ejemplo
  de balanceo específico de clase.
- Opción de crear datasets balanceados por muestreo entre clases.
"""

import tensorflow as tf
from modulos.lemon_dataset import LemonDataset


class LemonTFLoader(LemonDataset):
    """Loader basado en `tf.data` para el dataset de limones.

    Proporciona métodos para obtener datasets listos para entrenamiento
    (`get_datasets`) y para obtener datasets balanceados por muestreo
    (`get_sampling_datasets`).

    Args:
        img_size (tuple): Tamaño (alto, ancho) al que se redimensionan imágenes.
        batch_size (int): Tamaño de lote para batching en tf.data.
        mode (str): 'scratch' o 'transfer' — determina el preprocesado.
    """

    def __init__(self, img_size=(224, 224), batch_size=32, mode='scratch'):
        LemonDataset.__init__(self, mode=mode, loader="tf")
        self.img_size = img_size
        self.batch_size = batch_size
        # Genera los splits y la estructura interna `self.splits`
        self._create_splits()
        # Mapeo de clases a índices (útil para operaciones con one-hot)
        self.class_to_index = {"bad": 0, "empty": 1, "good": 2}
        self.mode = mode

    # Nota: la versión anterior de `_process_path` se eliminó porque
    # estaba duplicada y era sobrescrita por la implementación
    # más completa que aparece a continuación.

    def _process_path(self, file_path, label):
        """Lee una imagen desde disco y aplica el preprocesado adecuado.

        El preprocesado depende de `self.mode`:
        - 'scratch': normaliza dividiendo por 255.0
        - 'trasfer': aplica `self.preprocess_fn` (debe estar definida)

        Args:
            file_path: ruta a la imagen (string tensor o Python string)
            label: índice de clase entero

        Returns:
            tuple: (image_tensor, one_hot_label)
        """
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32)

        # Aplicar el preprocesado según el modo configurado
        if self.mode == "scratch":
            image = image / 255.0
        elif self.mode == "transfer":
            image = self.preprocess_fn(image)

        label = tf.one_hot(label, depth=3)
        return image, label

    def _augment(self, image, label):
        """Aplica transformaciones aleatorias sobre la imagen.

        Incluye flip horizontal, ajustes de brillo/contraste/saturación/tono y
        un recorte central aleatorio seguido de resize.
        """
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, self.max_delta)
        image = tf.image.random_contrast(
            image, self.contrast_range[0], self.contrast_range[1])
        image = tf.image.random_saturation(image, 0.6, 1.4)
        image = tf.image.random_hue(image, 0.08)
        crop_scale = tf.random.uniform([], 0.75, 0.95)
        image = tf.image.central_crop(image, crop_scale)
        image = tf.image.resize(image, self.img_size)
        return image, label

    def _conditional_augment(self, image, label):
        """Aplica augmentation condicionalmente según la etiqueta.

        Actualmente configurado para aplicar augmentation únicamente cuando
        la etiqueta corresponde a la clase `empty` (índice 1). La función
        usa `tf.cond` para evaluar la condición a nivel de grafo.
        """
        def augment():
            # Para activar augment real, descomentar la siguiente línea:
            # return self._augment(image, label)
            return image, label

        def no_augment():
            return image, label

        return tf.cond(
            tf.equal(tf.argmax(label), self.class_to_index['empty']),
            augment,
            no_augment
        )

    def get_datasets(self):
        """Crea y devuelve los datasets de entrenamiento, validación y prueba.

        Flujo básico:
        1. Convertir splits a `tf.data.Dataset`.
        2. Mapear con `_process_path` para leer/normalizar.
        3. Aplicar `_conditional_augment` sobre el train set.
        4. Shuffle, batch y prefetch.
        """
        train_ds = tf.data.Dataset.from_tensor_slices(self.splits["train"])
        val_ds = tf.data.Dataset.from_tensor_slices(self.splits["val"])
        test_ds = tf.data.Dataset.from_tensor_slices(self.splits["test"])

        train_ds = train_ds.map(
            self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(self._process_path,
                            num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(self._process_path,
                              num_parallel_calls=tf.data.AUTOTUNE)

        # Augmentation condicional (solo para muestras marcadas como 'empty')
        train_ds = train_ds.map(self._conditional_augment,
                                num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.shuffle(512).batch(
            self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Cardinalidad del dataset de entrenamiento (batches por epoch)
        try:
            self.steps_per_epoch = tf.data.experimental.cardinality(
                train_ds).numpy()
        except Exception:
            self.steps_per_epoch = None

        return train_ds, val_ds, test_ds

    def get_sampling_datasets(self):
        """Crea datasets balanceados por batch usando `sample_from_datasets`.

        Proceso:
        1. Construir un `train_raw` sin augment para luego filtrar por clase.
        2. Crear sub-datasets por clase y muestrearlos con pesos.
        3. Aplicar augment de forma condicional, luego batch y prefetch.
        """
        # ---------------------------
        # 1. Crear datasets base
        # ---------------------------
        train_raw = tf.data.Dataset.from_tensor_slices(self.splits["train"])
        val_ds = tf.data.Dataset.from_tensor_slices(self.splits["val"])
        test_ds = tf.data.Dataset.from_tensor_slices(self.splits["test"])

        # procesamiento simple (sin augment todavía)
        train_raw = train_raw.map(
            self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(self._process_path,
                            num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(self._process_path,
                              num_parallel_calls=tf.data.AUTOTUNE)

        # ---------------------------
        # 2. Subdividir por clase
        # ---------------------------
        ds_bad = train_raw.filter(lambda x, y: tf.equal(tf.argmax(y), 0))
        ds_empty = train_raw.filter(lambda x, y: tf.equal(tf.argmax(y), 1))
        ds_good = train_raw.filter(lambda x, y: tf.equal(tf.argmax(y), 2))

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

        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # calcular steps_per_epoch (estimación simple)
        self.steps_per_epoch = len(self.splits["train"]) // self.batch_size

        return train_ds, val_ds, test_ds
