"""
Evaluación del modelo entrenado (versión clase).

Proporciona `LemonEvaluator` para:
- reconstruir datasets de test SIN reentrenar
- cargar el modelo guardado (best / best_head / best_fine / best_overall)
- calcular métricas (matriz de confusión, classification_report)
- graficar y guardar la matriz de confusión (sns heatmap)
- visualizar imágenes mal clasificadas en grilla (4 columnas)
- exponer métodos amigables para usar desde Jupyter

Notas importantes:
- Si mode == "transfer", este Evaluator usa LemonTransferTrainer (Opción A),
  garantizando que el TFLoader reciba el preprocess_fn correcto por arquitectura.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from modulos.lemon_trainer import LemonTrainer, TrainerConfig

# Transfer trainer (opción A)
from modulos.transfer_trainer import LemonTransferTrainer


@dataclass
class EvalConfig:
    """Configuración para evaluación."""
    class_names: Tuple[str, ...] = ("bad", "empty", "good")
    cm_filename: str = "confusion_matrix.png"
    max_misclassified: int = 9


class LemonEvaluator:
    """Evaluador reutilizable para modelos entrenados con LemonTrainer / LemonTransferTrainer."""

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
        model_variant: str = "best",
        # --- Solo transfer ---
        architecture: str = "resnet50",
        fine_tune_at: int = 40,
    ):
        """
        Args:
            loader: "tf" o "gen"
            mode: "scratch" o "transfer"
            attempt: prefijo del modelo guardado
            trainer_cfg: config completa (si se pasa, loader/mode se ignoran)
            eval_cfg: configuración de evaluación
            val_size, test_size, seed: deben coincidir con el split del entrenamiento
            model_variant: "best", "best_overall", "best_head", "best_fine"
            architecture: (solo transfer) resnet50/xception/inceptionv3/mobilenetv2/densenet
            fine_tune_at: (solo transfer) parámetro que requiere LemonTransferTrainer (no afecta evaluación en sí)
        """
        self.eval_cfg = eval_cfg or EvalConfig()
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.seed = int(seed)

        self.trainer_cfg = trainer_cfg or TrainerConfig(loader=loader, mode=mode)
        self.attempt = attempt

        self.model_variant = model_variant

        # transfer params
        self.architecture = architecture
        self.fine_tune_at = int(fine_tune_at)

        # filled in prepare()
        self.trainer: Optional[object] = None
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
    def _collect_y_true_y_pred(model, dataset, verbose=True):
        """
        Soporta:
        - tf.data.Dataset finito
        - Keras generator/Sequence con __len__ (ImageDataGenerator)
        """
        # 1) Predicción
        preds = model.predict(dataset, verbose=1 if verbose else 0)
        y_pred = np.argmax(preds, axis=1)

        # 2) Etiquetas reales
        y_true = []

        # Caso generator/Sequence: no iterar infinitamente
        if hasattr(dataset, "__len__") and not isinstance(dataset, tf.data.Dataset):
            steps = len(dataset)
            if hasattr(dataset, "reset"):
                dataset.reset()

            it = iter(dataset)
            for _ in range(steps):
                _, labels = next(it)
                labels_np = labels.numpy() if hasattr(labels, "numpy") else labels
                if getattr(labels_np, "ndim", 1) > 1:
                    labels_np = np.argmax(labels_np, axis=1)
                y_true.extend(labels_np.tolist())

        else:
            # Caso tf.data.Dataset finito
            for _, labels in dataset:
                labels_np = labels.numpy() if hasattr(labels, "numpy") else labels
                if getattr(labels_np, "ndim", 1) > 1:
                    labels_np = np.argmax(labels_np, axis=1)
                y_true.extend(labels_np.tolist())

        return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    
    @staticmethod
    def _load_image_for_display(path: str, img_size) -> np.ndarray:
        """Carga imagen desde disco para visualizar (SIN preprocess_input)."""
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.uint8)  # display friendly
        return img.numpy()

    def _require_prepared(self):
        if self.trainer is None or self.model is None:
            raise RuntimeError("Primero llama a .prepare() para reconstruir datasets y cargar el modelo.")

    def _resolve_model_path(self) -> str:
        """
        Decide qué archivo cargar según model_variant.
        - scratch: usa trainer.best_model_path por defecto
        - transfer: respeta best_overall/best_head/best_fine según convención de LemonTransferTrainer
        """
        # best default (compatible con LemonTrainer y LemonTransferTrainer)
        if hasattr(self.trainer, "best_model_path"):
            model_path = self.trainer.best_model_path
        else:
            raise RuntimeError("Trainer no expone 'best_model_path'.")

        # si piden explícitos (transfer)
        if self.model_variant in ("best_overall", "best_head", "best_fine"):
            suffix = {
                "best_overall": "best_overall.keras",
                "best_head": "best_head.keras",
                "best_fine": "best_fine.keras",
            }[self.model_variant]

            save_dir = getattr(self.trainer, "save_dir", None)
            if save_dir is None:
                raise RuntimeError("Trainer no expone 'save_dir' para resolver model_variant.")

            name = f"{self.attempt}_{suffix}" if self.attempt else suffix
            candidate = os.path.join(save_dir, name)

            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f"No existe el modelo '{self.model_variant}' en: {candidate}\n"
                    f"Tip: asegúrate de que attempt='{self.attempt}' coincide con el entrenamiento."
                )
            model_path = candidate

        return model_path

    # -----------------------
    # Preparación principal
    # -----------------------
    def prepare(self) -> "LemonEvaluator":
        """Reconstruye datasets (sin reentrenar) y carga el modelo elegido."""

        # Opción A: usar el trainer correcto según el modo
        if self.trainer_cfg.mode == "transfer":
            self.trainer = LemonTransferTrainer(
                self.trainer_cfg,
                attempt=self.attempt,
                architecture=self.architecture,
                fine_tune_at=self.fine_tune_at,
            )
        else:
            self.trainer = LemonTrainer(self.trainer_cfg, attempt=self.attempt)

        # Importante: reconstruye splits de igual forma
        self.trainer.prepare_data(val_size=self.val_size, test_size=self.test_size, seed=self.seed)

        # cargar modelo
        model_path = self._resolve_model_path()
        self.model = self._safe_load_model(model_path)

        # reset cache
        self.y_true = None
        self.y_pred = None
        return self

    # -----------------------
    # Predicción y cache
    # -----------------------
    def predict(self, force: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula y cachea y_true/y_pred sobre test_ds."""
        self._require_prepared()
        if (self.y_true is None or self.y_pred is None) or force:
            self.y_true, self.y_pred = self._collect_y_true_y_pred(
                self.model, self.trainer.test_ds, verbose=verbose
            )
        return self.y_true, self.y_pred

    # -----------------------
    # Métricas
    # -----------------------
    def get_confusion_matrix(self, normalize: Optional[str] = None) -> np.ndarray:
        y_true, y_pred = self.predict()
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def get_classification_report(self, digits: int = 4) -> str:
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
        """Grafica (sns heatmap) y guarda la matriz de confusión en trainer.save_dir."""
        self._require_prepared()

        cm = self.get_confusion_matrix(normalize=normalize)

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
            save_dir = getattr(self.trainer, "save_dir", None)
            if save_dir is None:
                raise RuntimeError("Trainer no expone 'save_dir' para guardar figuras.")

            os.makedirs(save_dir, exist_ok=True)
            out_name = f"{self.attempt}_{self.eval_cfg.cm_filename}" if self.attempt else self.eval_cfg.cm_filename
            out_path = os.path.join(save_dir, out_name)
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
        """Muestra hasta `max_images` mal clasificadas en grilla 4 columnas."""
        self._require_prepared()
        max_images = int(max_images) if max_images is not None else int(self.eval_cfg.max_misclassified)

        y_true, y_pred = self.predict()

        wrong_idxs = np.where(y_true != y_pred)[0]
        total_wrong = int(len(wrong_idxs))
        print(f"Total mal clasificadas en TODO el test: {total_wrong}")

        if total_wrong == 0:
            return 0

        to_show = wrong_idxs[:max_images].tolist()
        class_names = list(self.eval_cfg.class_names)

        # -------------------------------------------------------
        # Intentar obtener paths del split test para mostrar "raw"
        # -------------------------------------------------------
        test_paths = None
        try:
            # En tu TFLoader, self.splits["test"] suele ser (paths, labels)
            loader = getattr(self.trainer, "loader", None)
            if loader is not None and hasattr(loader, "splits"):
                test_split = loader.splits.get("test")
                if isinstance(test_split, (tuple, list)) and len(test_split) >= 1:
                    test_paths = test_split[0]  # paths tensor/array
        except Exception:
            test_paths = None

        # ---- Configuración de grilla ----
        ncols = 4
        nimgs = len(to_show)
        nrows = int(np.ceil(nimgs / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)

        shown = 0
        for idx_global in to_show:
            r = shown // ncols
            c = shown % ncols
            ax = axes[r, c]

            # Mostrar imagen "bonita" (sin preprocess) si tenemos path
            if test_paths is not None:
                # test_paths puede ser np.array, list o tf.Tensor
                p = test_paths[idx_global]
                if isinstance(p, (bytes, bytearray)):
                    p = p.decode("utf-8")
                elif hasattr(p, "numpy"):
                    p = p.numpy().decode("utf-8")

                img = self._load_image_for_display(p, self.trainer.cfg.img_size)
            else:
                # Fallback: muestra el tensor del dataset (puede verse raro en transfer)
                # Recorremos hasta encontrar el idx_global (menos eficiente pero funciona)
                count = 0
                img = None
                for images, _labels in self.trainer.test_ds:
                    bs = int(images.shape[0])
                    if count + bs > idx_global:
                        img = images[idx_global - count].numpy()
                        if img.dtype != np.uint8:
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                        break
                    count += bs
                if img is None:
                    img = np.zeros((*self.trainer.cfg.img_size, 3), dtype=np.uint8)

            ax.imshow(img)
            ax.set_title(
                f"Idx {idx_global}\n"
                f"Real: {class_names[y_true[idx_global]]}\n"
                f"Pred: {class_names[y_pred[idx_global]]}",
                fontsize=9
            )
            ax.axis("off")
            shown += 1

        # Ocultar ejes vacíos
        for k in range(shown, nrows * ncols):
            axes[k // ncols, k % ncols].axis("off")

        plt.tight_layout()
        plt.show()

        return total_wrong