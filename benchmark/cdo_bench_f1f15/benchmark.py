from typing import Dict, List

from benchmark.cdo_bench_f1f14.benchmark import Benchmark as CDOBenchF1F14Benchmark
from benchmark._cdo_f15_shared.benchmark import MASOIEWSNFunction


class Benchmark:
    """
    Root benchmark-entry style CDO benchmark.

    F1-F14 reuse CDOBenchF1F14. F15 reuses the existing MASOIE WSN
    localization benchmark with its default first function.
    """

    def __init__(self, opts=None):
        self.opts = opts
        self._cdo = CDOBenchF1F14Benchmark(opts)
        self._wsn_f15 = None
        self.problem_ids = list(range(1, 16))

    def _get_wsn_f15(self) -> MASOIEWSNFunction:
        if self._wsn_f15 is None:
            seed = int(getattr(self.opts, "seed", 42)) if self.opts is not None else 42
            self._wsn_f15 = MASOIEWSNFunction(
                node_num=20,
                target_num=5,
                space_size=100.0,
                noise_std=2.0,
                seed=seed,
            )
        return self._wsn_f15

    def get_function(self, func_id: int):
        fid = int(func_id)
        if 1 <= fid <= 14:
            return self._cdo.get_function(fid)
        if fid == 15:
            return self._get_wsn_f15()
        raise ValueError("CDOBenchF1F15 function id is out of range. Available: 1..15.")

    def get_info(self, func_id: int) -> Dict:
        fid = int(func_id)
        if 1 <= fid <= 14:
            info = dict(self._cdo.get_info(fid))
            info["family"] = "CDOBenchF1F15"
            return info
        if fid == 15:
            info = dict(self._get_wsn_f15().info())
            info["family"] = "CDOBenchF1F15"
            info["source_family"] = "WSNLocationMASOIE"
            info["source_function_id"] = 1
            return info
        raise ValueError("CDOBenchF1F15 function id is out of range. Available: 1..15.")

    def get_num_functions(self) -> int:
        return len(self.problem_ids)

    def get_function_names(self) -> List[str]:
        return [f"CDO_Bench_F{i}" for i in range(1, 15)] + ["MASOIE_WSN_F15"]
