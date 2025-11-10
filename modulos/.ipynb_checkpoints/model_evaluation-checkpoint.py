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



