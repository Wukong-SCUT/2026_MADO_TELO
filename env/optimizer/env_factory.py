from env.optimizer.opt_ma import opt_ma
from env.optimizer.opt_ma_objective_split import opt_ma_cdo_objective, opt_ma_dbo_objective, opt_ma_masoie_wsn_objective
from env.optimizer.opt_ma_wsn_objective import opt_ma_wsn_objective


def make_opt_env(question: int, opts_in=None):
    mode = str(getattr(opts_in, "mappo_env_mode", "variable_cc")).lower()
    if mode == "wsn_objective":
        return opt_ma_wsn_objective(question, opts_in=opts_in)
    if mode == "dbo_objective":
        return opt_ma_dbo_objective(question, opts_in=opts_in)
    if mode == "cdo_objective":
        return opt_ma_cdo_objective(question, opts_in=opts_in)
    if mode == "masoie_wsn_objective":
        return opt_ma_masoie_wsn_objective(question, opts_in=opts_in)
    if mode == "variable_cc":
        return opt_ma(question, opts_in=opts_in)
    raise ValueError(f"Unsupported mappo_env_mode: {mode}")
