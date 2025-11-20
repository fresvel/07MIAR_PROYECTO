from __future__ import annotations
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from modulos.lemon_trainer import LemonTrainer, TrainerConfig
from modulos.lemon_tfloader import LemonTFLoader

# Modelos y preprocessors disponibles
from tensorflow.keras.applications import (
    ResNet50, Xception, InceptionV3, MobileNetV2, DenseNet121
)
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep
from tensorflow.keras.applications.xception import preprocess_input as xcep_prep
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_prep
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_prep
from tensorflow.keras.applications.densenet import preprocess_input as dense_prep


PREPROCESSORS = {
    "resnet50": resnet_prep,
    "xception": xcep_prep,
    "inceptionv3": inc_prep,
    "mobilenetv2": mob_prep,
    "densenet": dense_prep
}

BASE_MODELS = {
    "resnet50": ResNet50,
    "xception": Xception,
    "inceptionv3": InceptionV3,
    "mobilenetv2": MobileNetV2,
    "densenet": DenseNet121
}


class LemonTransferTrainer(LemonTrainer):
    """Trainer especializado en Transfer Learning + Fine Tuning.

    Extiende `LemonTrainer` y añade métodos para construir una cabeza
    clasificadora sobre un modelo base pre-entrenado, entrenar solo la
    cabeza y posteriormente realizar fine-tuning parcial.
    """

    def __init__(self, config: Optional[TrainerConfig] = None, attempt="",
                 architecture: str = "resnet50", fine_tune_at: int = 40):

        super().__init__(config, attempt)

        self.architecture = architecture.lower()
        if self.architecture not in BASE_MODELS:
            raise ValueError(f"Arquitectura desconocida '{architecture}'. Opciones: {list(BASE_MODELS.keys())}")

        # número de capas desde el final que se desbloquearán para fine-tuning
        self.fine_tune_at = int(fine_tune_at) if fine_tune_at is not None else 0
        self.prep_fn = PREPROCESSORS.get(self.architecture)

        # placeholders que se llenan en build_model
        self.base_model = None

    # ------------------------------------------------------
    # PREPARACIÓN DE DATOS (reusa prepare_data())
    # ------------------------------------------------------
    def prepare_data(self, val_size=0.15, test_size=0.15, seed=42):

        # Sobrescribimos solo el loader con preprocess_input si existe
        self.loader = LemonTFLoader(
            img_size=self.cfg.img_size,
            batch_size=self.cfg.batch_size,
            mode="transfer"
        )
        if self.prep_fn is not None:
            self.loader.preprocess_fn = self.prep_fn

        self.loader._create_splits(val_size=val_size, test_size=test_size, seed=seed)
        self.train_ds, self.val_ds, self.test_ds = self.loader.get_datasets()

        return self

    # ------------------------------------------------------
    #    1. Construir modelo base + Top classifier
    # ------------------------------------------------------
    def build_model(self):

        base_class = BASE_MODELS[self.architecture]
        base = base_class(
            weights="imagenet",
            include_top=False,
            input_shape=(self.cfg.img_size[0], self.cfg.img_size[1], 3)
        )
        base.trainable = False

        inputs = layers.Input(shape=(self.cfg.img_size[0], self.cfg.img_size[1], 3))
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu",
                         kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.cfg.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)
        self.base_model = base

        return self

    # ------------------------------------------------------
    #    2. FASE 1 — Entrenar solo top layers
    # ------------------------------------------------------
    def _train_head(self):

        self.model.compile(
            optimizer=optimizers.Adam(1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        cb = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.2)
        ]

        print("** Entrenando solo top layers **")
        steps = getattr(self.loader, "steps_per_epoch", None)
        return self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.cfg.epochs,
            callbacks=cb,
            steps_per_epoch=steps,
            verbose=1
        )

    # ------------------------------------------------------
    #    3. FASE 2 — Fine tuning parcial
    # ------------------------------------------------------
    def _train_finetune(self):

        if self.base_model is None:
            raise RuntimeError("El modelo base no está construido. Llama a build_model() antes de _train_finetune().")

        total_layers = len(self.base_model.layers)
        n = min(self.fine_tune_at, total_layers)
        if n <= 0:
            print("fine_tune_at <= 0, no se realizará fine-tuning.")
        else:
            for layer in self.base_model.layers[-n:]:
                layer.trainable = True

        self.model.compile(
            optimizer=optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        cb = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.2)
        ]

        print("** Fine Tuning activado **")
        steps = getattr(self.loader, "steps_per_epoch", None)
        return self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.cfg.epochs,
            callbacks=cb,
            steps_per_epoch=steps,
            verbose=1
        )

    # ------------------------------------------------------
    #   Sobrescribir método train()
    # ------------------------------------------------------
    def train(self):
        print("******** Training Transfer Model *******")
        print("Architecture:", self.architecture)
        steps = getattr(self.loader, "steps_per_epoch", None)
        print("Steps:", steps)

        # fase 1
        self.history_head = self._train_head()

        # fase 2
        self.history_fine = self._train_finetune()

        # El history principal será el del fine tuning
        self.history = self.history_fine
        return self


    # ------------------------------------------------------
    # run_trainer() se mantiene igual
    # ------------------------------------------------------
