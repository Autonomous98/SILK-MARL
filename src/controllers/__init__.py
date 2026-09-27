REGISTRY = {}

from .basic_controller import BasicMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
