# modulos/lemon_trainer.py
"""Entrenador unificado para el proyecto 07MIAR.

Proporciona una clase `LemonTrainer` que encapsula el flujo completo
de entrenamiento: preparación de datos (soporta `ImageDataGenerator` y
`tf.data`), construcción del modelo, compilación, entrenamiento,
evaluación y visualización del historial.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam

from modulos.lemon_cnn_model import LemonCNNBuilder
from modulos.lemon_genloader import LemonGenLoader
from modulos.lemon_tfloader import LemonTFLoader


@dataclass
class TrainerConfig:
    """Configuración del experimento para `LemonTrainer`.

    Atributos principales:
        - `loader`: 'gen' para ImageDataGenerator o 'tf' para tf.data.
        - `img_size`: tupla (alto, ancho) usada para redimensionar imágenes.
        - `batch_size`, `epochs`, `learning_rate`: parámetros estándar.
        - `mode`: 'scratch' o 'transfer' (afecta augmentation en loaders).
        - `model_out`: nombre de archivo para guardar el mejor modelo.
        - `patience_es`, `patience_rlrop`, `min_lr`: parámetros de callbacks.
        - `save_dir`: directorio base para resultados.
        - `num_classes`: número de clases de salida.
    """
    loader: str = "gen"           # "gen" → ImageDataGenerator, "tf" → tf.data
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    mode: str = "scratch"         # afecta augmentation
    model_out: str = "model_best.keras"
    patience_es: int = 20
    patience_rlrop: int = 8
    min_lr: float = 1e-6
    save_dir: str = "./results"
    num_classes: int = 3


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
        self.save_dir = os.path.join(
            self.cfg.save_dir, self.cfg.loader, attempt)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.save_dir, self.cfg.model_out)

    # -------------------------
    # PREPARACIÓN DE DATOS
    # -------------------------
    def prepare_data(self, val_size=0.15, test_size=0.15, seed=42):
        """Prepara los datos según `cfg.loader`.

        - Si `loader=='gen'` crea un `LemonGenLoader` y obtiene generators.
        - Si `loader=='tf'` crea un `LemonTFLoader`, genera splits y obtiene
          `tf.data.Dataset`.

        Devuelve `self` para permitir encadenado (`.prepare_data().build_model()`).
        """

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
            self.loader._create_splits(
                val_size=val_size, test_size=test_size, seed=seed)
            self.train_ds, self.val_ds, self.test_ds = self.loader.get_datasets()

        else:
            raise ValueError(
                f"loader '{self.cfg.loader}' no reconocido. Use 'gen' o 'tf'.")

        return self

    # -------------------------
    # MODELO
    # -------------------------
    def build_model(self):
        """Construye la arquitectura del modelo usando `LemonCNNBuilder`.

        No compila el modelo; llame a `compile_model()` para compilarlo.
        """
        self.builder = LemonCNNBuilder(
            input_shape=(self.cfg.img_size[0], self.cfg.img_size[1], 3),
            num_classes=self.cfg.num_classes
        )
        self.model = self.builder.build()
        return self

    def compile_model(self):
        """Compila el modelo con Adam y pérdida categórica.

        Ajusta métricas a `accuracy`.
        """
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
        """Devuelve los callbacks usados durante el entrenamiento.

        Incluye EarlyStopping, ReduceLROnPlateau y ModelCheckpoint.
        """
        return [
            EarlyStopping(
                monitor="val_loss", patience=self.cfg.patience_es, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                              patience=self.cfg.patience_rlrop, min_lr=self.cfg.min_lr),
            ModelCheckpoint(self.best_model_path,
                            monitor="val_loss", save_best_only=True)
        ]

    # -------------------------
    # TRAIN
    # -------------------------
    def train(self):
        """Entrena el modelo usando los datasets/generadores preparados.

        Usa `steps_per_epoch` del loader si está definido; Keras calculará
        automáticamente los pasos si se pasa `None`.
        """
        print("******** Training model *******")
        print("Loader:", self.cfg.loader)
        steps = getattr(self.loader, "steps_per_epoch", None)
        print("Steps:", steps)

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.cfg.epochs,
            callbacks=self._callbacks(),
            verbose=1,
            steps_per_epoch=steps
        )
        return self

    # -------------------------
    # EVALUATE
    # -------------------------
    def evaluate(self) -> Dict[str, Any]:
        """Evalúa el modelo en el conjunto de prueba y devuelve métricas.

        Retorna un diccionario con `test_loss`, `test_accuracy` y la ruta
        al mejor modelo guardado.
        """
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
        """Dibuja y guarda curvas del entrenamiento (loss y métricas)."""
        if self.history is None:
            raise RuntimeError("No hay 'history'. Entrena el modelo primero con .train().")

        hist = pd.DataFrame(self.history.history)

        # Normalizar nombres típicos (acc -> accuracy) si hiciera falta
        if "acc" in hist.columns and "accuracy" not in hist.columns:
            hist = hist.rename(columns={"acc": "accuracy"})
        if "val_acc" in hist.columns and "val_accuracy" not in hist.columns:
            hist = hist.rename(columns={"val_acc": "val_accuracy"})

        # ---- 1) LOSS ----
        loss_cols = [c for c in ["loss", "val_loss"] if c in hist.columns]
        if loss_cols:
            ax = hist[loss_cols].plot(
                figsize=(12, 5),
                grid=True,
                xlabel="Época",
                linestyle=["-", "--"][:len(loss_cols)]
            )
            ax.set_ylabel("Loss")
            ax.figure.tight_layout()
            ax.figure.savefig(os.path.join(self.save_dir, "history_loss.png"))
            plt.show()

        # ---- 2) ACCURACY (u otra métrica principal) ----
        metric_cols = [c for c in ["accuracy", "val_accuracy"] if c in hist.columns]
        if metric_cols:
            ax = hist[metric_cols].plot(
                figsize=(12, 5),
                grid=True,
                xlabel="Época",
                ylim=[0, 1],
                linestyle=["-", "--"][:len(metric_cols)]
            )
            ax.set_ylabel("Accuracy")
            ax.figure.tight_layout()
            ax.figure.savefig(os.path.join(self.save_dir, "history_acc.png"))
            plt.show()

        # Si quieres, también puedes guardar el CSV del history para trazabilidad:
        # hist.to_csv(os.path.join(self.save_dir, "history.csv"), index=False)

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
