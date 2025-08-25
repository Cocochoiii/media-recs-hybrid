from dataclasses import dataclass

@dataclass
class TrainConfig:
    factors: int = 64
    lr: float = 1e-2
    weight_decay: float = 1e-5
    epochs: int = 5
    neg_ratio: int = 4
