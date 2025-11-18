import numpy as np
import os

from modulos.transfer_trainer import LemonTransferTrainer
from modulos.lemon_tfloader import LemonTFLoader
from modulos.lemon_trainer import TrainerConfig   # se reutiliza el mismo dataclass


# ===========================================================
# CONFIGURACIÓN GENERAL DEL EXPERIMENTO
# ===========================================================

RUNS = 10

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

ARCHITECTURE = "MobileNetV2"     # Cambia aquí para probar otra red: "xception", "inceptionv3", etc.
FINE_TUNE_AT = 40             # Cuántas capas descongelar en el fine tuning

SAVE_DIR = "./results_transfer/" + ARCHITECTURE
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------------------------------------
# Crear configuración base del trainer
# -----------------------------------------------------------
cfg = TrainerConfig(
    loader="tf",            # usamos tf.data
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    epochs=30,
    learning_rate=1e-4,     # usada en la fase head
    mode="transfer",        # activación del loader para transfer learning
    save_dir=SAVE_DIR
)


# ===========================================================
# LISTAS PARA GUARDAR MÉTRICAS ENTRE CORRIDAS
# ===========================================================

test_accs = []
val_accs = []


# ===========================================================
# EJECUTAR 10 ENTRENAMIENTOS COMPLETOS
# ===========================================================

for i in range(1, RUNS + 1):

    attempt_id = f"{i:02d}"
    print(f"\n================ INICIO DEL ENTRENAMIENTO {attempt_id} =================")

    # Crear trainer
    trainer = LemonTransferTrainer(
        cfg,
        attempt=attempt_id,
        architecture=ARCHITECTURE,
        fine_tune_at=FINE_TUNE_AT
    )

    # Ejecutar entrenamiento completo
    results = trainer.run_trainer()

    # Accuracy test
    test_acc = results["test_accuracy"]
    test_accs.append(test_acc)

    # Accuracy final en validación (última época del fine tuning)
    final_val_acc = trainer.history.history["val_accuracy"][-1]
    val_accs.append(final_val_acc)

    print(f"   → Test accuracy       : {test_acc:.4f}")
    print(f"   → Final Val accuracy  : {final_val_acc:.4f}")


# ===========================================================
# ESTADÍSTICAS FINALES ENTRE CORRIDAS
# ===========================================================

test_mean = np.mean(test_accs)
test_std  = np.std(test_accs)
test_var  = np.var(test_accs)

val_mean = np.mean(val_accs)
val_std  = np.std(val_accs)
val_var  = np.var(val_accs)

print("\n================ VARIANZA ENTRE CORRIDAS (TRANSFER) =================")
print(f"Test Accuracy: mean={test_mean:.4f}, var={test_var:.6f}, std={test_std:.6f}")
print(f"Val Accuracy : mean={val_mean:.4f}, var={val_var:.6f}, std={val_std:.6f}")
print("======================================================================\n")


# ===========================================================
# GUARDAR RESULTADOS EN ARCHIVOS .npy
# ===========================================================

np.save(os.path.join(SAVE_DIR, "test_accs.npy"), np.array(test_accs))
np.save(os.path.join(SAVE_DIR, "val_accs.npy"),  np.array(val_accs))

print("Resultados guardados en:", SAVE_DIR)
print("Archivos creados:")
print("   - test_accs.npy")
print("   - val_accs.npy")
print("   - val_accs.npy")
print("\nTodo listo.\n")
