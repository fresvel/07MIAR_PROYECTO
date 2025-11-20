"""run_transfer.py

Script de entrada para ejecutar múltiples corridas de transfer learning
usando `LemonTransferTrainer`.

Este script está pensado como ejemplo/experimento: ejecuta `RUNS`
entrenamientos completos, guarda las métricas (test / val) en arrays
NumPy y escribe los archivos `.npy` en un subdirectorio por
arquitectura.

Cambios realizados:
- Encapsulado en `main()` y agregado `if __name__ == '__main__'`.
- Logging en lugar de `print` para mensajes informativos y errores.
- Validaciones sobre `results` y `trainer.history` para evitar
  excepciones si faltan claves o atributos.
- Corrección menor: eliminación de print duplicado.
"""

from pathlib import Path
import logging
import numpy as np
import os
from typing import Optional

from modulos.transfer_trainer import LemonTransferTrainer
from modulos.lemon_trainer import TrainerConfig   # se reutiliza el mismo dataclass


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s - %(message)s",
    )


def main(
    runs: int = 10,
    img_size: tuple = (224, 224),
    batch_size: int = 32,
    architecture: str = "MobileNetV2",
    fine_tune_at: int = 40,
    epochs: int = 30,
    learning_rate: float = 1e-4,
):
    """Ejecuta `runs` experimentos de transfer learning y guarda métricas.

    Parámetros:
    - runs: número de corridas independientes a ejecutar.
    - img_size, batch_size: configuración del loader / trainer.
    - architecture: nombre de la arquitectura para `LemonTransferTrainer`.
    - fine_tune_at: número de capas a descongelar en la fase de fine-tune.
    - epochs, learning_rate: parámetros del entrenamiento de la cabeza.
    """

    _setup_logging()
    log = logging.getLogger(__name__)

    save_dir = Path("results_transfer") / architecture
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

    test_accs = []
    val_accs = []

    for i in range(1, runs + 1):
        attempt_id = f"{i:02d}"
        log.info("INICIO DEL ENTRENAMIENTO %s", attempt_id)

        trainer = LemonTransferTrainer(
            cfg,
            attempt=attempt_id,
            architecture=architecture,
            fine_tune_at=fine_tune_at,
        )

        try:
            results = trainer.run_trainer()
        except Exception as e:
            log.exception("Error al ejecutar run_trainer() en intento %s", attempt_id)
            # Mantener el ciclo; registrar NaN para mantener las longitudes
            test_accs.append(np.nan)
            val_accs.append(np.nan)
            continue

        # Recuperar test accuracy de forma segura
        test_acc: Optional[float] = None
        if isinstance(results, dict) and "test_accuracy" in results:
            try:
                test_acc = float(results["test_accuracy"])
            except Exception:
                test_acc = np.nan
        else:
            log.warning("Resultados sin 'test_accuracy' para intento %s", attempt_id)
            test_acc = np.nan

        test_accs.append(test_acc)

        # Recuperar última val_accuracy de trainer.history si existe
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
            log.warning("trainer.history no disponible para intento %s", attempt_id)

        val_accs.append(final_val_acc)

        log.info("Test accuracy: %.4f | Final val accuracy: %.4f", float(test_acc), float(final_val_acc))

    # Estadísticas finales
    test_arr = np.array(test_accs, dtype=float)
    val_arr = np.array(val_accs, dtype=float)

    def _stats(arr: np.ndarray):
        return np.nanmean(arr), np.nanvar(arr), np.nanstd(arr)

    test_mean, test_var, test_std = _stats(test_arr)
    val_mean, val_var, val_std = _stats(val_arr)

    log.info("VARIANZA ENTRE CORRIDAS (TRANSFER)")
    log.info("Test Accuracy: mean=%.4f, var=%.6f, std=%.6f", test_mean, test_var, test_std)
    log.info("Val  Accuracy: mean=%.4f, var=%.6f, std=%.6f", val_mean, val_var, val_std)

    # Guardar resultados
    np.save(save_dir / "test_accs.npy", test_arr)
    np.save(save_dir / "val_accs.npy", val_arr)

    log.info("Resultados guardados en: %s", str(save_dir))


if __name__ == "__main__":
    # Valores por defecto; el usuario puede modificar la llamada si desea otras configuraciones
    main(
        runs=10,
        img_size=(224, 224),
        batch_size=32,
        architecture="MobileNetV2",
        fine_tune_at=40,
        epochs=30,
        learning_rate=1e-4,
    )
