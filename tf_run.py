import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from modulos.lemon_trainer import LemonTrainer, TrainerConfig

cfg = TrainerConfig(
    loader="tf",        # tf o "gen"
    mode="scratch",
    epochs=40,
    learning_rate=1e-3
)

trainer = LemonTrainer(cfg)
results = trainer.run_trainer()
results
