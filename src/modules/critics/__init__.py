from .centralV import CentralVCritic
from .maddpg import MADDPGCritic


REGISTRY = {}

REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["maddpg_critic"] = MADDPGCritic
