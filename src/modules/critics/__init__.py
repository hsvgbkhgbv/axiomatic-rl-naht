from .centralV import CentralVCritic
from .centralV_ns import CentralVCriticNS
from .ac import ACCritic
from .ac_ns import ACCriticNS
from .ippo import IPPOCritic
from .ippo_ns import IPPOCriticNS
from .poam import POAMCritic
from .poam_shapley1 import POAMShapleyCritic1

REGISTRY = {}

REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["cv_critic_ns"] = CentralVCriticNS
REGISTRY["ac_critic"] = ACCritic
REGISTRY["ac_critic_ns"] = ACCriticNS
REGISTRY["ippo_critic"] = IPPOCritic
REGISTRY["ippo_critic_ns"] = IPPOCriticNS
REGISTRY["poam_critic"] = POAMCritic
REGISTRY["poam_shapley_critic1"] = POAMShapleyCritic1


