# Local optimizers

The E3aj MAPPO policy selects among four local optimizers: `mmes`, `vkd`, `cmaes`, and `sepcmaes`. Their common factory is `optimizers/unified_opt/factory.py`; implementations are in the same directory. Shared CMA-ES components are in `optimizers/cmaes/`.

The policy also selects an optimizer profile and a resource factor. The packaged defaults are defined in `release.py` and listed in `configs/algorithm.json`; the training-specific values are listed in `configs/training_e3aj.json`. The optimizers are called by the CDO/WSN environments; `python release.py cdo` and `python release.py wsn` are the supported evaluation entry points.
