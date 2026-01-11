"""run_transfer.py

Script de entrada para ejecutar múltiples corridas de transfer learning
usando `LemonTransferTrainer`, repitiendo el experimento para múltiples
arquitecturas (ResNet50, Xception, InceptionV3, MobileNetV2, DenseNet121).

Guarda:
- results_transfer/<arch>/test_accs.npy
- results_transfer/<arch>/val_accs.npy

y (si está implementado en LemonTransferTrainer):
- modelos best_* y gráficas dentro de results_transfer/<arch>/tf/
"""

from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from modulos.transfer_trainer import LemonTransferTrainer
from modulos.lemon_trainer import TrainerConfig


# Arquitecturas objetivo (claves coherentes con LemonTransferTrainer)
ARCHS: Tuple[str, ...] = ("resnet50", "xception", "inceptionv3", "mobilenetv2", "densenet")


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s - %(message)s",
    )


def _stats(arr: np.ndarray):
    return np.nanmean(arr), np.nanvar(arr), np.nanstd(arr)


def run_for_architecture(
    arch_key: str,
    runs: int,
    img_size: tuple,
    batch_size: int,
    fine_tune_at: int,
    epochs: int,
    learning_rate: float,
    base_save_dir: Path,
) -> Dict[str, np.ndarray]:
    """Ejecuta `runs` corridas para una arquitectura específica y guarda .npy."""
    log = logging.getLogger(__name__)

    save_dir = base_save_dir / arch_key
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainerConfig(
        loader="tf",
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        mode="transfer",
        save_dir=str(save_dir),
    )

    test_accs: List[float] = []
    val_accs: List[float] = []

    log.info("==============================================")
    log.info("ARQUITECTURA: %s", arch_key)
    log.info("Resultados en: %s", str(save_dir))
    log.info("==============================================")

    for i in range(1, runs + 1):
        attempt_id = f"{i:02d}"
        log.info("[%s] INICIO DEL ENTRENAMIENTO %s", arch_key, attempt_id)

        trainer = LemonTransferTrainer(
            cfg,
            attempt=attempt_id,
            architecture=arch_key,   # clave en minúsculas
            fine_tune_at=fine_tune_at,
        )

        try:
            results = trainer.run_trainer()
        except Exception:
            log.exception("[%s] Error al ejecutar run_trainer() en intento %s", arch_key, attempt_id)
            test_accs.append(np.nan)
            val_accs.append(np.nan)
            continue

        # Test acc
        test_acc: Optional[float] = np.nan
        if isinstance(results, dict) and "test_accuracy" in results:
            try:
                test_acc = float(results["test_accuracy"])
            except Exception:
                test_acc = np.nan
        else:
            log.warning("[%s] Resultados sin 'test_accuracy' para intento %s", arch_key, attempt_id)

        test_accs.append(test_acc)

        # Final val acc desde history (fine)
        final_val_acc = np.nan
        hist = getattr(trainer, "history", None)
        if hist is not None and hasattr(hist, "history"):
            val_list = hist.history.get("val_accuracy") if isinstance(hist.history, dict) else None
            if val_list:
                try:
                    final_val_acc = float(val_list[-1])
                except Exception:
                    final_val_acc = np.nan
        else:
            log.warning("[%s] trainer.history no disponible para intento %s", arch_key, attempt_id)

        val_accs.append(final_val_acc)

        log.info("[%s] Test acc: %.4f | Final val acc: %.4f", arch_key, float(test_acc), float(final_val_acc))

    # Guardar arrays
    test_arr = np.array(test_accs, dtype=float)
    val_arr = np.array(val_accs, dtype=float)

    np.save(save_dir / "test_accs.npy", test_arr)
    np.save(save_dir / "val_accs.npy", val_arr)

    # Logs estadísticos
    test_mean, test_var, test_std = _stats(test_arr)
    val_mean, val_var, val_std = _stats(val_arr)

    log.info("[%s] VARIANZA ENTRE CORRIDAS (TRANSFER)", arch_key)
    log.info("[%s] Test Acc: mean=%.4f, var=%.6f, std=%.6f", arch_key, test_mean, test_var, test_std)
    log.info("[%s] Val  Acc: mean=%.4f, var=%.6f, std=%.6f", arch_key, val_mean, val_var, val_std)
    log.info("[%s] Resultados guardados en: %s", arch_key, str(save_dir))

    return {"test_accs": test_arr, "val_accs": val_arr}


def main(
    runs: int = 10,
    img_size: tuple = (224, 224),
    batch_size: int = 32,
    fine_tune_at: int = 40,
    epochs: int = 30,
    learning_rate: float = 1e-4,
    architectures: Tuple[str, ...] = ARCHS,
):
    """Ejecuta el experimento para varias arquitecturas."""

    # Reproducibilidad (ojo: algunos ops GPU siguen siendo no deterministas)
    tf.keras.utils.set_random_seed(42)

    _setup_logging()
    log = logging.getLogger(__name__)

    base_save_dir = Path("results_transfer")
    base_save_dir.mkdir(parents=True, exist_ok=True)

    log.info("===== TRANSFER LEARNING MULTI-ARQUITECTURA =====")
    log.info("Runs=%d | img_size=%s | batch=%d | fine_tune_at=%d | epochs=%d | lr=%g",
             runs, img_size, batch_size, fine_tune_at, epochs, learning_rate)
    log.info("Arquitecturas: %s", ", ".join(architectures))
    log.info("Directorio base: %s", str(base_save_dir))

    for arch_key in architectures:
        run_for_architecture(
            arch_key=arch_key,
            runs=runs,
            img_size=img_size,
            batch_size=batch_size,
            fine_tune_at=fine_tune_at,
            epochs=epochs,
            learning_rate=learning_rate,
            base_save_dir=base_save_dir,
        )

    log.info("===== FIN EXPERIMENTOS TRANSFER MULTI-ARQ =====")


if __name__ == "__main__":
    main(
        runs=10,
        img_size=(224, 224),
        batch_size=32,
        fine_tune_at=40,
        epochs=30,
        learning_rate=1e-4,
        architectures=ARCHS,
    )
