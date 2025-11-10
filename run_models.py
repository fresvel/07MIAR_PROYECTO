from modulos.lemon_trainer import LemonTrainer, TrainerConfig

cfg = TrainerConfig(
    loader="gen",        # o "gen"
    mode="scratch",
    epochs=40,
    learning_rate=1e-3
)

trainer = LemonTrainer(cfg)
results = trainer.run_trainer()
results
