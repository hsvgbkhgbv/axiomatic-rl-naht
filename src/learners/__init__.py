from .q_learner import QLearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .ppo_learner import PPOLearner
from .poam_learner import POAMLearner
from .poam_shapley_learner import POAMShapleyLearner
from .ppo_shapley_learner import PPOShapleyLearner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["poam_learner"] = POAMLearner
REGISTRY["poam_shapley_learner"] = POAMShapleyLearner
REGISTRY["ppo_shapley_learner"] = PPOShapleyLearner