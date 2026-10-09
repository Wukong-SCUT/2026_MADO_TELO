from env.optimizer.opt_ma_objective_split import opt_ma_cdo_objective
from env.optimizer.opt_ma_wsn_objective import opt_ma_wsn_objective


def make_opt_env(question: int, opts_in=None):
    mode = str(getattr(opts_in, "mappo_env_mode", "cdo_objective")).lower()
    if mode == "wsn_objective":
        return opt_ma_wsn_objective(question, opts_in=opts_in)
    if mode == "cdo_objective":
        return opt_ma_cdo_objective(question, opts_in=opts_in)
    raise ValueError(f"Unsupported mappo_env_mode: {mode}")
