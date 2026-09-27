from .q_learner import QLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner


REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
