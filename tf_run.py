"""tf_run.py

Script ejemplo para ejecutar múltiples entrenamientos en modo "scratch"
usando `LemonTrainer`.

El script fue refactorizado para:
- Estar encapsulado en `main()` y ejecutarse con `if __name__ == '__main__'`.
- Usar `logging` en lugar de `print` para mensajes informativos.
- Manejar errores en `run_trainer()` y validar que `results` y
  `trainer.history` contienen los valores esperados.
- Guardar métricas finales en `results_tf/` como archivos `.npy`.
"""

from pathlib import Path
import logging
from typing import Optional

import numpy as np

from modulos.lemon_trainer import LemonTrainer, TrainerConfig

import tensorflow as tf

def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")


def main(
    runs: int = 10,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    loader: str = "tf",
    mode: str = "scratch",
):
    """Ejecuta `runs` entrenamientos en modo scratch y guarda métricas.

    Parámetros:
    - runs: número de corridas a ejecutar.
    - epochs, learning_rate: parámetros de entrenamiento.
    - loader, mode: pasan al `TrainerConfig`.
    """

    tf.keras.utils.set_random_seed(42)

    _setup_logging()
    log = logging.getLogger(__name__)

    cfg = TrainerConfig(loader=loader, mode=mode, epochs=epochs, learning_rate=learning_rate)

    test_accs = []
    val_accs = []

    save_dir = Path("results_tf")
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, runs + 1):
        attempt_id = f"{i:02d}"
        log.info("Iniciando entrenamiento intento %s", attempt_id)

        trainer = LemonTrainer(cfg, attempt=attempt_id)

        try:
            results = trainer.run_trainer()
        except Exception:
            log.exception("run_trainer falló en intento %s", attempt_id)
            test_accs.append(np.nan)
            val_accs.append(np.nan)
            continue

        # Obtener test_accuracy de forma segura
        test_acc: Optional[float] = None
        if isinstance(results, dict) and "test_accuracy" in results:
            try:
                test_acc = float(results["test_accuracy"])
            except Exception:
                test_acc = np.nan
        else:
            log.warning("No se encontró 'test_accuracy' en results del intento %s", attempt_id)
            test_acc = np.nan

        test_accs.append(test_acc)

        # Obtener val_accuracy desde trainer.history si está disponible
        val_acc = np.nan
        hist = getattr(trainer, "history", None)
        if hist is not None and hasattr(hist, "history"):
            val_list = hist.history.get("val_accuracy") if isinstance(hist.history, dict) else None
            if val_list:
                try:
                    val_acc = float(val_list[-1])
                except Exception:
                    val_acc = np.nan
        else:
            log.warning("trainer.history no disponible para intento %s", attempt_id)

        val_accs.append(val_acc)

        log.info("Test acc: %.4f | Val acc final: %.4f", float(test_acc), float(val_acc))

    # Estadísticas finales (ignorar NaNs)
    test_arr = np.array(test_accs, dtype=float)
    val_arr = np.array(val_accs, dtype=float)

    def _stats(arr: np.ndarray):
        return float(np.nanmean(arr)), float(np.nanvar(arr)), float(np.nanstd(arr))

    test_mean, test_var, test_std = _stats(test_arr)
    val_mean, val_var, val_std = _stats(val_arr)

    log.info("VARIANZA ENTRE CORRIDAS")
    log.info("Test Accuracy: mean=%.4f, var=%.6f, std=%.6f", test_mean, test_var, test_std)
    log.info("Val  Accuracy: mean=%.4f, var=%.6f, std=%.6f", val_mean, val_var, val_std)

    # Guardar resultados
    np.save(save_dir / "test_accs.npy", test_arr)
    np.save(save_dir / "val_accs.npy", val_arr)

    log.info("Resultados guardados en %s", str(save_dir))


if __name__ == "__main__":
    main()
