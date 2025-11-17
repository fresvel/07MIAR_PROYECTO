import numpy as np
from modulos.lemon_trainer import LemonTrainer, TrainerConfig

# Configuración base
cfg = TrainerConfig(
    loader="tf",
    mode="scratch",
    epochs=40,
    learning_rate=1e-3
)

# Guardar métricas de cada corrida
test_accs = []
val_accs = []

# Ejecutar 10 intentos de entrenamiento
for i in range(1, 11):
    attempt_id = f"{i:02d}"
    print(f"\n===== Iniciando entrenamiento intento {attempt_id} =====")

    trainer = LemonTrainer(cfg, attempt=attempt_id)
    results = trainer.run_trainer()

    test_acc = results["test_accuracy"]
    test_accs.append(test_acc)

    # Obtener accuracy final del validation set (tomada del history)
    val_acc = trainer.history.history["val_accuracy"][-1]
    val_accs.append(val_acc)

    print(f"Test accuracy: {test_acc:.4f} | Val accuracy final: {val_acc:.4f}")

# --------------------------------------------------------------------
# Cálculo estadístico de la varianza entre corridas
# --------------------------------------------------------------------
test_mean = np.mean(test_accs)
test_var  = np.var(test_accs)
test_std  = np.std(test_accs)

val_mean = np.mean(val_accs)
val_var  = np.var(val_accs)
val_std  = np.std(val_accs)

print("\n================ VARIANZA ENTRE CORRIDAS ================")
print(f"Test Accuracy:    mean={test_mean:.4f}, var={test_var:.6f}, std={test_std:.6f}")
print(f"Val Accuracy:     mean={val_mean:.4f},  var={val_var:.6f},  std={val_std:.6f}")
print("==========================================================\n")
