# modulos/lemon_trainer.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from modulos.lemon_tfloader import LemonTFLoader
from modulos.lemon_genloader import LemonGenLoader
from modulos.lemon_cnn_model import LemonCNNBuilder


# ============================================================
# CONFIGURACIÓN BASE
# ============================================================
@dataclass
class TrainerConfig:
    loader: str = "gen"           # "gen" → ImageDataGenerator, "tf" → tf.data
    img_size: Tuple[int,int] = (224,224)
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    mode: str = "scratch"         # afecta augmentación
    model_out: str = "model_best.keras"
    patience_es: int = 20
    patience_rlrop: int = 8
    min_lr: float = 1e-6
    save_dir: str = "./results"
    num_classes: int = 3


# ============================================================
# ENTRENADOR UNIFICADO
# ============================================================
class LemonTrainer:
    """
    Clase unificada para entrenamiento.
    Decide automáticamente si usar ImageDataGenerator o tf.data según config.loader.
    Mantiene constantes el modelo, el optimizador y la lógica experimental.
    """
    def __init__(self, config: Optional[TrainerConfig] = None, attempt=""):
        self.cfg = config or TrainerConfig()
        self.loader = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.builder = None
        self.model = None
        self.history = None

        # Crear carpeta de salida
        self.save_dir = os.path.join(self.cfg.save_dir, self.cfg.loader, attempt)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.save_dir, self.cfg.model_out)

    # ----------------------------------------------------------
    # PREPARACIÓN DE DATOS
    # ----------------------------------------------------------
    def prepare_data(self, val_size=0.15, test_size=0.15, seed=42):
        if self.cfg.loader == "gen":
            self.loader = LemonGenLoader(
                img_size=self.cfg.img_size,
                batch_size=self.cfg.batch_size,
                mode=self.cfg.mode
            )
            self.train_ds, self.val_ds, self.test_ds = self.loader.get_generators()

        elif self.cfg.loader == "tf":
            self.loader = LemonTFLoader(
                img_size=self.cfg.img_size,
                batch_size=self.cfg.batch_size,
                mode=self.cfg.mode
            )
            self.loader._create_splits(val_size=val_size, test_size=test_size, seed=seed)
            self.train_ds, self.val_ds, self.test_ds = self.loader.get_datasets()

        else:
            raise ValueError(f"loader '{self.cfg.loader}' no reconocido. Use 'gen' o 'tf'.")

        return self

    # ----------------------------------------------------------
    # MODELO
    # ----------------------------------------------------------
    def build_model(self):
        self.builder = LemonCNNBuilder(
            input_shape=(self.cfg.img_size[0], self.cfg.img_size[1], 3),
            num_classes=self.cfg.num_classes
        )
        self.model = self.builder.build()
        return self

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(self.cfg.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return self

    # ----------------------------------------------------------
    # CALLBACKS
    # ----------------------------------------------------------
    def _callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", patience=self.cfg.patience_es, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=self.cfg.patience_rlrop, min_lr=self.cfg.min_lr),
            ModelCheckpoint(self.best_model_path, monitor="val_loss", save_best_only=True)
        ]

    # ----------------------------------------------------------
    # TRAIN con PONDERACIÓN DE CLASES
    # ----------------------------------------------------------
    def train(self):
        print("******** Training model *******")
        print("Loader:", self.cfg.loader)

        # ------------------------------------------------------
        # Calcular ponderación de clases
        # ------------------------------------------------------
        if self.cfg.loader == "tf":
            _, labels = self.loader.splits["train"]
            labels = labels.numpy().tolist() if isinstance(labels, tf.Tensor) else list(labels)
        else:
            labels = self.loader._train_df["class"].map({"bad": 0, "empty": 1, "good": 2}).values

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )

        alpha = 0.5         # suaviza a la raíz cuadrada
        max_ratio = 2.5     # no permitas que ninguna clase pese >2.5x otra

        w = np.power(class_weights, alpha)
        w = np.minimum(w, np.max(w)/np.minimum.reduce([1,1]) )  # opcional
        # o normaliza por el máximo para limitar el rango
        w = w / np.max(w) * max_ratio

        class_weight_dict = dict(enumerate(class_weights))
        print("Class weights:", class_weight_dict)

        # ------------------------------------------------------
        # Entrenamiento principal
        # ------------------------------------------------------
        use_class_weight = self.cfg.loader == "tf"

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.cfg.epochs,
            callbacks=self._callbacks(),
            verbose=1,
            steps_per_epoch=getattr(self.loader, "steps_per_epoch", None),
            **({"class_weight": class_weight_dict} if use_class_weight else {})
        )
        return self


    # ----------------------------------------------------------
    # EVALUACIÓN FINAL
    # ----------------------------------------------------------
    def evaluate(self) -> Dict[str, Any]:
        test_loss, test_acc = self.model.evaluate(self.test_ds, verbose=0)
        return {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "best_model_path": self.best_model_path
        }

    # ----------------------------------------------------------
    # VISUALIZACIÓN
    # ----------------------------------------------------------
    def plot_history(self):
        if self.history is None:
            raise RuntimeError("No hay 'history'. Entrena el modelo primero con .train().")

        hist = self.history.history
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(hist["loss"], label="Train Loss")
        plt.plot(hist["val_loss"], label="Val Loss")
        plt.title("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(hist["accuracy"], label="Train Acc")
        plt.plot(hist["val_accuracy"], label="Val Acc")
        plt.title("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "history.png"))
        plt.show()
        return self

    # ----------------------------------------------------------
    # EXPERIMENTO COMPLETO
    # ----------------------------------------------------------
    def run_trainer(self, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42) -> Dict[str, Any]:
        """
        Ejecuta el flujo completo del entrenamiento:
        preparación de datos, compilación, entrenamiento, evaluación y visualización.
        """
        (self.prepare_data(val_size=val_size, test_size=test_size, seed=seed)
            .build_model()
            .compile_model()
            .train())
        metrics = self.evaluate()
        self.plot_history()
        return metrics
