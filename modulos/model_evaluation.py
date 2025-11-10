##Cargar el modelo

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from modulos.lemon_trainer import LemonTrainer, TrainerConfig

# -- 1) Reconstruimos el trainer para obtener los datasets (sin reentrenar)
cfg = TrainerConfig(loader="tf", mode="scratch")   # usa el mismo loader que entrenaste
trainer = LemonTrainer(cfg)
trainer.prepare_data()  # esto reconstruye train/val/test
trainer.build_model()

# -- 2) Cargar el mejor modelo guardado
model = tf.keras.models.load_model(trainer.best_model_path)

# -- 3) Obtener predicciones
y_true = []
y_pred = []

for images, labels in trainer.test_ds:
    preds = model.predict(images, verbose=0)

    y_true.extend(tf.argmax(labels, axis=1).numpy())
    y_pred.extend(tf.argmax(preds, axis=1).numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("✅ Datos recogidos para evaluación.")


##Matriz de confusión y reporte

from sklearn.metrics import confusion_matrix, classification_report

class_names = ["bad", "empty", "good"]

print("Matriz de Confusión:\n")
print(confusion_matrix(y_true, y_pred))

print("\nReporte de Clasificación:\n")
print(classification_report(y_true, y_pred, target_names=class_names))


### visualizar imágenes mal clasificadas


import matplotlib.pyplot as plt

def show_misclassified(model, dataset, class_names):
    for images, labels in dataset.take(1):
        preds = model.predict(images, verbose=0)
        y_true = tf.argmax(labels, axis=1)
        y_pred = tf.argmax(preds, axis=1)
        
        wrong = tf.where(y_true != y_pred).numpy().flatten()

        print(f"Total mal clasificadas en este batch: {len(wrong)}")
        for idx in wrong[:9]:  # mostrar máximo 9
            plt.figure(figsize=(2,2))
            plt.imshow(images[idx])
            plt.title(f"Real: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
            plt.axis("off")
            plt.show()

show_misclassified(model, trainer.test_ds, class_names)




