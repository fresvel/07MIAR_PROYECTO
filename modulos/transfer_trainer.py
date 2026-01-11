from __future__ import annotations
from typing import Optional, Dict

import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from modulos.lemon_trainer import LemonTrainer, TrainerConfig
from modulos.lemon_tfloader import LemonTFLoader

# --------------------------------------------------
# Modelos base y preprocessors
# --------------------------------------------------
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
    "densenet": dense_prep,
}

BASE_MODELS = {
    "resnet50": ResNet50,
    "xception": Xception,
    "inceptionv3": InceptionV3,
    "mobilenetv2": MobileNetV2,
    "densenet": DenseNet121,
}


class LemonTransferTrainer(LemonTrainer):
    """
    Trainer para Transfer Learning + Fine Tuning.

    Fases:
    1) HEAD: modelo base congelado, se entrena solo la cabeza
    2) FINE: se descongelan capas finales y se afina el modelo

    Modelos guardados:
    - best_head.keras
    - best_fine.keras
    - best_overall.keras  (alias explícito del mejor modelo final)
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        attempt: str = "",
        architecture: str = "resnet50",
        fine_tune_at: int = 40,
    ):
        super().__init__(config, attempt)

        self.architecture = architecture.lower()
        if self.architecture not in BASE_MODELS:
            raise ValueError(
                f"Arquitectura desconocida '{architecture}'. "
                f"Opciones: {list(BASE_MODELS.keys())}"
            )

        self.fine_tune_at = int(fine_tune_at)
        self.prep_fn = PREPROCESSORS.get(self.architecture)

        self.base_model = None

        # ---------------- RUTAS DE GUARDADO ----------------
        self.best_head_path = os.path.join(
            self.save_dir, f"{self.attempt}_best_head.keras"
        )
        self.best_fine_path = os.path.join(
            self.save_dir, f"{self.attempt}_best_fine.keras"
        )
        self.best_overall_path = os.path.join(
            self.save_dir, f"{self.attempt}_best_overall.keras"
        )

        self.history_head = None
        self.history_fine = None

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------
    def prepare_data(self, val_size=0.15, test_size=0.15, seed=42):
        self.loader = LemonTFLoader(
            img_size=self.cfg.img_size,
            batch_size=self.cfg.batch_size,
            mode="transfer",
        )
        if self.prep_fn is not None:
            self.loader.preprocess_fn = self.prep_fn

        self.loader._create_splits(
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )
        self.train_ds, self.val_ds, self.test_ds = self.loader.get_datasets()
        return self

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    def build_model(self):
        base_cls = BASE_MODELS[self.architecture]
        base = base_cls(
            weights="imagenet",
            include_top=False,
            input_shape=(*self.cfg.img_size, 3),
        )
        base.trainable = False

        inputs = layers.Input(shape=(*self.cfg.img_size, 3))
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.cfg.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)
        self.base_model = base
        return self

    # --------------------------------------------------
    # PHASE 1 — HEAD
    # --------------------------------------------------
    def _train_head(self):
        self.model.compile(
            optimizer=optimizers.Adam(1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                patience=3,
                factor=0.2,
            ),
            ModelCheckpoint(
                self.best_head_path,
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        print("** FASE 1 — Entrenando HEAD **")
        return self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.cfg.epochs,
            callbacks=callbacks,
            verbose=1,
        )

    # --------------------------------------------------
    # PHASE 2 — FINE TUNING
    # --------------------------------------------------
    def _train_finetune(self):
        if self.base_model is None:
            raise RuntimeError("build_model() debe ejecutarse antes de fine-tuning.")

        total_layers = len(self.base_model.layers)
        n = min(self.fine_tune_at, total_layers)

        for layer in self.base_model.layers[-n:]:
            layer.trainable = True

        self.model.compile(
            optimizer=optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                patience=4,
                factor=0.2,
            ),
            ModelCheckpoint(
                self.best_fine_path,
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        print("** FASE 2 — Fine Tuning **")
        return self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.cfg.epochs,
            callbacks=callbacks,
            verbose=1,
        )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    def train(self):
        print("******** Transfer Learning ********")
        print("Architecture:", self.architecture)
        print("Save dir:", self.save_dir)

        self.history_head = self._train_head()
        self.history_fine = self._train_finetune()

        # El history principal queda asociado al fine
        self.history = self.history_fine

        # ---------------- BEST OVERALL ----------------
        if os.path.exists(self.best_fine_path):
            shutil.copy2(self.best_fine_path, self.best_overall_path)
            print("✔ best_overall guardado desde best_fine")

        return self

    # --------------------------------------------------
    # UTILS
    # --------------------------------------------------
    def saved_best_paths(self) -> Dict[str, str]:
        return {
            "best_head": self.best_head_path,
            "best_fine": self.best_fine_path,
            "best_overall": self.best_overall_path,
        }
