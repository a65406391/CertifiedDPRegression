"""Optimization-baesd Abstract Gradient Training (AGT)."""

import logging

from abstract_gradient_training_opt import certified_training
from abstract_gradient_training_opt import configuration
from abstract_gradient_training_opt import training_utils
from abstract_gradient_training_opt.configuration import OPTConfig
from abstract_gradient_training_opt.formulations import functional
from abstract_gradient_training_opt.formulations import modules
from abstract_gradient_training_opt.formulations import polytope
from abstract_gradient_training_opt.formulations import variables

logger = logging.getLogger("abstract_gradient_training_opt")
logger.handlers.clear()
formatter = logging.Formatter(
    "[OPT] [%(levelname)-8s] [%(asctime)s] %(message)s", datefmt="%H:%M:%S"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

__all__ = [
    "certified_training",
    "configuration",
    "functional",
    "modules",
    "variables",
    "polytope",
    "OPTConfig",
    "training_utils",
]
