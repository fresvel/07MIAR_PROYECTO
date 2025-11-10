# modulos/lemon_trainer.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

from modulos.lemon_tfloader import LemonTFLoader
from modulos.lemon_genloader import LemonGenLoader
from modulos.lemon_cnn_model import LemonCNNBuilder

@dataclass
class TrainerConfig:
    loader: str = "gen"           # "gen" → ImageDataGenerator, "tf" → tf.data
    img_size: Tuple[int,int] = (224,224)
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    mode: str = "scratch"         # afecta augmentación
    model_out: str = "model_best.keras"
    patience_es: int = 7
    patience_rlrop: int = 3
    min_lr: float = 1e-6
    save_dir: str = "./results"
    num_classes: int = 3


class LemonTrainer:
    """
    Clase unificada para entrenamiento.
    Decide automáticamente si usar ImageDataGenerator o tf.data según config.loader.
    Mantiene constantes el modelo, el optimizador y la lógica experimental.
    """
    def __init__(self, config: Optional[TrainerConfig] = None):
        self.cfg = config or TrainerConfig()
        self.loader = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.builder = None
        self.model = None
        self.history = None
        self.cfg.save_dir = os.path.join(self.cfg.save_dir, self.cfg.loader)
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.cfg.save_dir, self.cfg.model_out)

    # -------------------------
    # PREPARACIÓN DE DATOS
    # -------------------------
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

    # -------------------------
    # MODELO
    # -------------------------
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

    # -------------------------
    # CALLBACKS
    # -------------------------
    def _callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", patience=self.cfg.patience_es, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=self.cfg.patience_rlrop, min_lr=self.cfg.min_lr),
            ModelCheckpoint(self.best_model_path, monitor="val_loss", save_best_only=True)
        ]

    # -------------------------
    # TRAIN
    # -------------------------
    def train(self):
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.cfg.epochs,
            callbacks=self._callbacks(),
            verbose=1
        )
        return self

    # -------------------------
    # EVALUATE
    # -------------------------
    def evaluate(self) -> Dict[str, Any]:
        test_loss, test_acc = self.model.evaluate(self.test_ds, verbose=0)
        return {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "best_model_path": self.best_model_path
        }

    # -------------------------
    # VISUALIZACIÓN
    # -------------------------
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
        plt.title("Accuracy"); plt.legend()
        plt.savefig(os.path.join(self.cfg.save_dir, "history.png"))
        plt.show()
        return self

    # -------------------------
    # RUN FULL EXPERIMENT
    # -------------------------
    def run_trainer(self, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42) -> Dict[str, Any]:
        """
        Ejecuta automáticamente todo el flujo experimental del entrenamiento supervisado,
        desde la preparación del conjunto de datos hasta la obtención de métricas finales.

        Parámetros
        ----------
        val_size : float, opcional (default=0.15)
            Proporción del dataset total que se asigna al conjunto de validación.
        
        test_size : float, opcional (default=0.15)
            Proporción del dataset total que se reserva para el conjunto de prueba.
        
        seed : int, opcional (default=42)
            Semilla para asegurar reproducibilidad en la división estratificada del dataset.

        Flujo de ejecución
        ------------------
        1) **prepare_data()**
        - Divide el dataset en train/validation/test manteniendo balance de clases.
        - Crea los generadores o datasets según la estrategia configurada (GenLoader o TFLoader).
        
        2) **build_model()**
        - Construye la arquitectura CNN propuesta mediante API Funcional.
        - Ajusta número de filtros, bloques convolucionales y capa densa final con softmax.
        
        3) **compile_model()**
        - Compila el modelo con optimizador Adam, función de pérdida categorical_crossentropy
            y la métrica accuracy.
        
        4) **train()**
        - Ejecuta el proceso de entrenamiento monitoreando val_loss.
        - Utiliza callbacks: EarlyStopping, ReduceLROnPlateau y ModelCheckpoint.
        - Guarda automáticamente el mejor modelo según desempeño en validación.
        
        5) **evaluate()**
        - Evalúa el modelo entrenado en el conjunto de prueba (test), retornando métricas finales.

        6) **plot_history()**
        - Genera visualizaciones de las curvas de pérdida y accuracy (entrenamiento vs validación),
            útiles para el análisis del comportamiento del modelo.

        Retorno
        -------
        dict
            Un diccionario con las métricas finales y la ruta donde se guardó el modelo con mejor desempeño.
            Ejemplo:
            {
                "test_loss": 0.34,
                "test_accuracy": 0.89,
                "best_model_path": "model_scratch_best.keras"
            }

        Descripción general
        -------------------
        Esta función abstrae y automatiza el flujo completo del experimento, facilitando la comparación entre:
        - Distintas configuraciones del modelo
        - Diferentes estrategias de entrenamiento (desde cero vs. transfer learning)
        - Distintos pipelines de carga (ImageDataGenerator vs. tf.data)

        De esta manera se asegura consistencia, reproducibilidad y organización en el proceso de experimentación.
        """
        (self.prepare_data(val_size=val_size, test_size=test_size, seed=seed)
            .build_model()
            .compile_model()
            .train())
        metrics = self.evaluate()
        self.plot_history()
        return metrics


