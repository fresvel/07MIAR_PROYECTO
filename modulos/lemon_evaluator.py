"""
Evaluación del modelo entrenado (versión clase).

Proporciona `LemonEvaluator` para:
- reconstruir datasets de test SIN reentrenar
- cargar el mejor modelo guardado (best_model_path)
- calcular métricas (matriz de confusión, classification_report)
- graficar y guardar la matriz de confusión (sns heatmap)
- visualizar imágenes mal clasificadas
- exponer métodos amigables para usar desde Jupyter

Uso típico en Jupyter:

from modulos.lemon_evaluator import LemonEvaluator

ev = LemonEvaluator(loader="tf", mode="scratch", attempt="")
ev.prepare()                 # reconstruye test_ds y carga modelo
cm = ev.confusion_matrix()
rep = ev.classification_report()
ev.plot_confusion_matrix()   # guarda en el mismo directorio de resultados
ev.show_misclassified(max_images=9)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from modulos.lemon_trainer import LemonTrainer, TrainerConfig


@dataclass
class EvalConfig:
    """Configuración para evaluación.

    - class_names: nombres de clases en el orden de índices (0..K-1)
    - cm_filename: nombre del archivo de salida para la matriz de confusión
    - max_misclassified: máximo de imágenes a mostrar en show_misclassified()
    """
    class_names: Tuple[str, ...] = ("bad", "empty", "good")
    cm_filename: str = "confusion_matrix.png"
    max_misclassified: int = 9


class LemonEvaluator:
    """Evaluador reutilizable para modelos entrenados con LemonTrainer.

    La idea es poder llamarlo desde Jupyter para:
    - preparar datasets y modelo una vez
    - consultar resultados con métodos específicos
    """

    def __init__(
        self,
        loader: str = "tf",
        mode: str = "scratch",
        attempt: str = "",
        trainer_cfg: Optional[TrainerConfig] = None,
        eval_cfg: Optional[EvalConfig] = None,
        val_size: float = 0.15,
        test_size: float = 0.15,
        seed: int = 42,
    ):
        """
        Args:
            loader: "tf" o "gen" (recomendado "tf" para evaluación del módulo actual).
            mode: "scratch" o "transfer" (debe coincidir con el entrenamiento si afecta al loader).
            attempt: prefijo del modelo guardado (misma convención que LemonTrainer).
            trainer_cfg: si quieres pasar un TrainerConfig completo. Si se pasa, loader/mode se ignoran.
            eval_cfg: configuración de evaluación (nombres de clases, nombres de archivos, etc).
            val_size, test_size, seed: deben coincidir con el entrenamiento si quieres el mismo split.
        """
        self.eval_cfg = eval_cfg or EvalConfig()
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.seed = int(seed)

        # Config del trainer: si no viene, creamos una mínima con loader/mode
        self.trainer_cfg = trainer_cfg or TrainerConfig(loader=loader, mode=mode)
        self.attempt = attempt

        # Se llenan en prepare()
        self.trainer: Optional[LemonTrainer] = None
        self.model: Optional[tf.keras.Model] = None
        self.y_true: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None

    # -----------------------
    # Utilidades
    # -----------------------
    @staticmethod
    def _safe_load_model(path: str) -> tf.keras.Model:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado en: {path}")
        return tf.keras.models.load_model(path)

    @staticmethod
    def _collect_y_true_y_pred(model: tf.keras.Model, dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_true: List[int] = []
        y_pred: List[int] = []

        for images, labels in dataset:
            preds = model.predict(images, verbose=0)

            true_idxs = tf.argmax(labels, axis=1).numpy()
            pred_idxs = tf.argmax(preds, axis=1).numpy()

            y_true.extend(true_idxs.tolist())
            y_pred.extend(pred_idxs.tolist())

        return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)

    def _require_prepared(self):
        if self.trainer is None or self.model is None:
            raise RuntimeError("Primero llama a .prepare() para reconstruir datasets y cargar el modelo.")

    # -----------------------
    # Preparación principal
    # -----------------------
    def prepare(self) -> "LemonEvaluator":
        """Reconstruye datasets (sin reentrenar) y carga el mejor modelo.

        - Crea un LemonTrainer con el mismo cfg (loader/mode).
        - Reconstruye el split con (val_size, test_size, seed).
        - Carga el modelo desde trainer.best_model_path.
        """
        self.trainer = LemonTrainer(self.trainer_cfg, attempt=self.attempt)
        self.trainer.prepare_data(val_size=self.val_size, test_size=self.test_size, seed=self.seed)
        self.model = self._safe_load_model(self.trainer.best_model_path)

        # reset cache de predicciones, por si se llama prepare() otra vez
        self.y_true = None
        self.y_pred = None
        return self

    def predict(self, force: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula y cachea y_true/y_pred sobre test_ds.

        Args:
            force: si True, recalcula aunque ya existan valores cacheados.

        Returns:
            (y_true, y_pred)
        """
        self._require_prepared()
        if (self.y_true is None or self.y_pred is None) or force:
            self.y_true, self.y_pred = self._collect_y_true_y_pred(self.model, self.trainer.test_ds)
        return self.y_true, self.y_pred

    # -----------------------
    # Métricas
    # -----------------------
    def confusion_matrix(self, normalize: Optional[str] = None) -> np.ndarray:
        """Devuelve la matriz de confusión.

        Args:
            normalize: None, "true", "pred", "all" (misma opción de sklearn)
        """
        y_true, y_pred = self.predict()
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def classification_report(self, digits: int = 4) -> str:
        """Devuelve el reporte de clasificación (texto)."""
        y_true, y_pred = self.predict()
        return classification_report(
            y_true,
            y_pred,
            target_names=list(self.eval_cfg.class_names),
            digits=digits
        )

    # -----------------------
    # Gráficas
    # -----------------------
    def plot_confusion_matrix(
        self,
        normalize: Optional[str] = None,
        cmap: str = "Blues",
        save: bool = True,
        show: bool = True,
        dpi: int = 300,
    ) -> Dict[str, Any]:
        """Grafica (sns heatmap) y guarda la matriz de confusión en trainer.save_dir.

        Args:
            normalize: None, "true", "pred", "all"
            cmap: mapa de color para heatmap
            save: si True, guarda PNG en el directorio de resultados
            show: si True, muestra la figura
            dpi: dpi de guardado

        Returns:
            dict con { "cm": array, "path": ruta_salida (o None) }
        """
        self._require_prepared()

        cm = self.confusion_matrix(normalize=normalize)

        sns.set_theme(style="whitegrid")
        sns.set_palette("Set2")

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        fmt = ".2f" if normalize else "d"
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            cbar=True,
            square=True,
            xticklabels=self.eval_cfg.class_names,
            yticklabels=self.eval_cfg.class_names,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
        )

        title = "Matriz de Confusión"
        if normalize:
            title += f" (normalize='{normalize}')"
        ax.set_title(title)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta real")
        plt.tight_layout()

        out_path = None
        if save:
            os.makedirs(self.trainer.save_dir, exist_ok=True)
            out_name = f"{self.attempt}_{self.eval_cfg.cm_filename}" if self.attempt else self.eval_cfg.cm_filename
            out_path = os.path.join(self.trainer.save_dir, out_name)

            
            plt.savefig(out_path, dpi=dpi)
            

        if show:
            plt.show()
        else:
            plt.close(fig)

        return {"cm": cm, "path": out_path}

    # -----------------------
    # Visualización cualitativa
    # -----------------------
    def show_misclassified(self, max_images: Optional[int] = None) -> int:
        """Muestra hasta `max_images` imágenes mal clasificadas en TODO el test.

        Retorna el total de mal clasificadas encontradas en todo el dataset (no solo en un batch).
        """
        self._require_prepared()
        max_images = int(max_images) if max_images is not None else int(self.eval_cfg.max_misclassified)

        # Asegurar predicciones globales (y_true/y_pred para TODO el test)
        y_true, y_pred = self.predict()

        wrong_idxs = np.where(y_true != y_pred)[0]
        total_wrong = int(len(wrong_idxs))
        print(f"Total mal clasificadas en TODO el test: {total_wrong}")

        if total_wrong == 0:
            return 0

        # Tomar solo las primeras N para mostrar
        to_show = wrong_idxs[:max_images].tolist()
        to_show_set = set(to_show)

        class_names = list(self.eval_cfg.class_names)

        # Recorremos el dataset manteniendo un índice global
        global_i = 0
        shown = 0

        for images, labels in self.trainer.test_ds:
            # Convertir etiquetas del batch a índices
            batch_true = tf.argmax(labels, axis=1).numpy()

            # Predicciones del batch
            batch_preds = self.model.predict(images, verbose=0)
            batch_pred = tf.argmax(batch_preds, axis=1).numpy()

            batch_size = images.shape[0]

            for j in range(batch_size):
                idx_global = global_i + j
                if idx_global in to_show_set:
                    img = images[j]
                    if hasattr(img, "numpy"):
                        img = img.numpy()

                    # Normalizar para mostrar
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)

                    plt.figure(figsize=(3, 3))
                    plt.imshow(img)
                    plt.title(
                        f"Idx: {idx_global}\nReal: {class_names[batch_true[j]]}\nPred: {class_names[batch_pred[j]]}"
                    )
                    plt.axis("off")
                    plt.show()

                    shown += 1
                    if shown >= len(to_show):
                        return total_wrong

            global_i += batch_size

        return total_wrong
