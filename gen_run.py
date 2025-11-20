"""Script de ejemplo para ejecutar un entrenamiento con `ImageDataGenerator`.

Este script crea una configuración mínima usando `TrainerConfig` con
`loader='gen'` (ImageDataGenerator), construye un `LemonTrainer` y
lanza el flujo completo de entrenamiento llamando a `run_trainer()`.

Uso:
    python gen_run.py

Nota: envolver la ejecución en `if __name__ == '__main__'` evita que el
entrenamiento se ejecute al importar el módulo desde otros scripts.
"""

from modulos.lemon_trainer import LemonTrainer, TrainerConfig
import logging


def main() -> dict:
    """Crea el trainer con configuración por defecto y ejecuta el flujo.

    Returns:
        dict: métricas devueltas por `run_trainer()`.
    """
    cfg = TrainerConfig(
        loader="gen",        # 'tf' o 'gen'
        mode="scratch",
        epochs=40,
        learning_rate=1e-3
    )

    trainer = LemonTrainer(cfg)
    results = trainer.run_trainer()
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        metrics = main()
        logging.info("Entrenamiento finalizado. Métricas: %s", metrics)
    except Exception as e:
        logging.exception("Error durante el entrenamiento: %s", e)
