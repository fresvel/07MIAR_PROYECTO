"""Evaluación del modelo entrenado.

Este script reconstruye el `LemonTrainer` para obtener los datasets
de test (sin reentrenar), carga el mejor modelo guardado y calcula la
matriz de confusión y el reporte de clasificación. También incluye una
función para visualizar imágenes mal clasificadas.

Uso: ejecutar este módulo dentro del repo (asume que los paths del
dataset y el mejor modelo están disponibles).
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from modulos.lemon_trainer import LemonTrainer, TrainerConfig
import matplotlib.pyplot as plt


def safe_load_model(path: str):
    """Carga un modelo Keras comprobando previamente que el archivo existe.

    Args:
        path: ruta al modelo (.keras / carpeta de SavedModel)

    Returns:
        tf.keras.Model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo no encontrado en: {path}")
    return tf.keras.models.load_model(path)


def evaluate_model_on_dataset(model, dataset):
    """Calcula etiquetas verdaderas y predichas sobre un dataset.

    Devuelve dos arrays numpy: y_true, y_pred (enteros de clase).
    """
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        # Soportar tanto tensores como arrays
        true_idxs = tf.argmax(labels, axis=1).numpy()
        pred_idxs = tf.argmax(preds, axis=1).numpy()

        y_true.extend(true_idxs.tolist())
        y_pred.extend(pred_idxs.tolist())

    return np.array(y_true), np.array(y_pred)


def show_misclassified(model, dataset, class_names, max_images=9):
    """Muestra hasta `max_images` imágenes mal clasificadas del primer batch.

    Maneja tanto tensores de TF como arrays de NumPy al mostrar imágenes.
    """
    for images, labels in dataset.take(1):
        preds = model.predict(images, verbose=0)
        y_true = tf.argmax(labels, axis=1).numpy()
        y_pred = tf.argmax(preds, axis=1).numpy()

        wrong = np.where(y_true != y_pred)[0]
        print(f"Total mal clasificadas en este batch: {len(wrong)}")

        for idx in wrong[:max_images]:
            img = images[idx]
            # convertir tensor a numpy si es necesario
            if hasattr(img, "numpy"):
                img = img.numpy()

            # Normalizar para mostrar (0-255 uint8)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            plt.figure(figsize=(3, 3))
            plt.imshow(img)
            plt.title(f"Real: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    # -- 1) Reconstruimos el trainer para obtener los datasets (sin reentrenar)
    cfg = TrainerConfig(loader="tf", mode="scratch")   # usar el mismo loader que en entrenamiento
    trainer = LemonTrainer(cfg)
    trainer.prepare_data()  # esto reconstruye train/val/test

    # -- 2) Cargar el mejor modelo guardado
    model = safe_load_model(trainer.best_model_path)

    # -- 3) Obtener predicciones y métricas
    y_true, y_pred = evaluate_model_on_dataset(model, trainer.test_ds)
    print("✅ Datos recogidos para evaluación.")

    class_names = ["bad", "empty", "good"]

    print("Matriz de Confusión:\n")
    print(confusion_matrix(y_true, y_pred))

    print("\nReporte de Clasificación:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # -- 4) Visualizar mal clasificadas
    show_misclassified(model, trainer.test_ds, class_names)




