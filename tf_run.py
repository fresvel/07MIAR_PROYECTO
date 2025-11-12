from modulos.lemon_trainer import LemonTrainer, TrainerConfig

# Configuración base
cfg = TrainerConfig(
    loader="tf",        # o "gen"
    mode="scratch",
    epochs=40,
    learning_rate=1e-3
)

# Ejecutar 10 intentos de entrenamiento
for i in range(1, 11):
    attempt_id = f"{i:02d}"  # genera 01, 02, ..., 10
    print(f"\n===== Iniciando entrenamiento intento {attempt_id} =====")

    trainer = LemonTrainer(cfg, attempt=attempt_id)
    results = trainer.run_trainer()

    print("************************************************************************")
    print(f"Entrenamiento {attempt_id} finalizado. Resultados obtenidos:")
    print(results)
    print("************************************************************************\n")
