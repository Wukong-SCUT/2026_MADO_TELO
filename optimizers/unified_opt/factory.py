from .mmes import MMES
from .vkd import VKD
from .cmaes_opt import CMAESOpt
from .sepcmaes_opt import SepCMAESOpt


def create_optimizer(name: str, problem, options):
    key = str(name).strip().lower()
    if key == "mmes":
        return MMES(problem, options)
    if key == "vkd":
        return VKD(problem, options)
    if key == "cmaes":
        return CMAESOpt(problem, options)
    if key == "sepcmaes":
        return SepCMAESOpt(problem, options)
    raise ValueError(f"Unsupported optimizer: {name}. Available: mmes, vkd, cmaes, sepcmaes")
