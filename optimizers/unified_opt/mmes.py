import numpy as np  # engine for numerical computing
from scipy.stats import norm  # normal continuous random variable

from .es import ES  # abstract class of all Evolution Strategies (ES) classes


class MMES(ES):
    """Mixture Model-based Evolution Strategy (MMES).

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'm'             - number of candidate direction vectors (`int`, default:
                  `2*int(np.ceil(np.sqrt(problem['ndim_problem'])))`),
                * 'c_c'           - learning rate of evolution path update (`float`, default:
                  `0.4/np.sqrt(problem['ndim_problem'])`),
                * 'ms'            - mixing strength (`int`, default: `4`),
                * 'c_s'           - learning rate of global step-size adaptation (`float`, default: `0.3`),
                * 'a_z'           - target significance level (`float`, default: `0.05`),
                * 'distance'      - minimal distance of updating evolution paths (`int`, default:
                  `int(np.ceil(1.0/options['c_c']))`),
                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the black-box optimizer `MMES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.mmes import MMES
       >>> problem = {'fitness_function': rosenbrock,  # to define problem arguments
       ...            'ndim_problem': 200,
       ...            'lower_boundary': -5.0*numpy.ones((200,)),
       ...            'upper_boundary': 5.0*numpy.ones((200,))}
       >>> options = {'max_function_evaluations': 500000,  # to set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3.0*numpy.ones((200,)),
       ...            'sigma': 3.0}  # global step-size may need to be tuned for optimality
       >>> mmes = MMES(problem, options)  # to initialize the optimizer class
       >>> results = mmes.optimize()  # to run the optimization/evolution process
       >>> print(f"MMES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       MMES: 500000, 2.6018

    For its correctness checking of Python coding, please refer to `this code-based repeatability report
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/_repeat_mmes.py>`_
    for all details. For *pytest*-based automatic testing, please see `test_mmes.py
    <https://github.com/Evolutionary-Intelligence/pypop/blob/main/pypop7/optimizers/es/test_mmes.py>`_.

    Attributes
    ----------
    a_z           : `float`
                    target significance level.
    c_c           : `float`
                    learning rate of evolution path update.
    c_s           : `float`
                    learning rate of global step-size adaptation.
    distance      : `int`
                    minimal distance of updating evolution paths.
    m             : `int`
                    number of candidate direction vectors.
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    ms            : `int`
                    mixing strength.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    He, X., Zheng, Z. and Zhou, Y., 2021.
    `MMES: Mixture model-based evolution strategy for large-scale optimization.
    <https://ieeexplore.ieee.org/abstract/document/9244595>`_
    IEEE Transactions on Evolutionary Computation, 25(2), pp.320-333.

    Please refer to the *official* Matlab version from Prof. He:
    https://github.com/hxyokokok/MMES
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        # set number of candidate direction vectors
        self.m = options.get('m', 2*int(np.ceil(np.sqrt(self.ndim_problem))))
        assert self.m > 0
        # set learning rate of evolution path
        self.c_c = options.get('c_c', 0.4/np.sqrt(self.ndim_problem))
        self.ms = options.get('ms', 4)  # mixing strength (l)
        assert self.ms > 0
        # set for paired test adaptation (PTA)
        self.c_s = options.get('c_s', 0.3)  # learning rate of global step-size adaptation
        self.a_z = options.get('a_z', 0.05)  # target significance level
        # set minimal distance of updating evolution paths (T)
        self.distance = options.get('distance', int(np.ceil(1.0/self.c_c)))
        # set success probability of geometric distribution (different from 4/n in the original paper)
        self.c_a = float(options.get('c_a', 3.8/self.ndim_problem))  # same as the official Matlab code
        # Numerical safety:
        # geometric(p) requires p in (0,1]. For very small subproblems (e.g., ndim=1/2/3),
        # default 3.8/ndim can exceed 1 and crash.
        self.c_a = float(np.clip(self.c_a, 1e-8, 1.0 - 1e-8))
        self.gamma = float(options.get('gamma', 1.0 - np.power(1.0 - self.c_a, self.m)))
        self.gamma = float(np.clip(self.gamma, 1e-12, 1.0 - 1e-12))
        self._n_mirror_sampling = None
        self._z_1 = np.sqrt(1.0 - self.gamma)
        self._z_2 = np.sqrt(self.gamma/self.ms)
        self._p_1 = 1.0 - self.c_c
        self._p_2 = np.sqrt(self.c_c*(2.0 - self.c_c))
        self._w_1 = 1.0 - self.c_s
        self._w_2 = np.sqrt(self.c_s*(2.0 - self.c_s))
        # Numerical guard for non-finite objective values.
        self.nonfinite_penalty = float(options.get("nonfinite_penalty", 1e300))

    def _repair_bounds(self, x: np.ndarray) -> np.ndarray:
        if (self.lower_boundary is None) or (self.upper_boundary is None):
            return x
        lb = np.asarray(self.lower_boundary, dtype=np.float64).reshape(-1)
        ub = np.asarray(self.upper_boundary, dtype=np.float64).reshape(-1)
        return np.clip(x, lb, ub)

    def _sanitize_fitness_batch(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if np.all(np.isfinite(y)):
            return y
        y = np.nan_to_num(
            y,
            nan=self.nonfinite_penalty,
            posinf=self.nonfinite_penalty,
            neginf=-self.nonfinite_penalty,
        )
        return y

    def initialize(self, args=None, is_restart=False):
        self._n_mirror_sampling = int(np.ceil(self.n_individuals/2))
        x = np.zeros((self.n_individuals, self.ndim_problem))  # offspring population
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p = np.zeros((self.ndim_problem,))  # evolution path
        w = 0.0
        q = np.zeros((self.m, self.ndim_problem))  # candidate direction vectors
        t = np.zeros((self.m,))  # recorded generations
        v = np.arange(self.m)  # indexes to evolution paths
        y = np.tile(self._evaluate_fitness(mean, args), (self.n_individuals,))  # fitness
        return x, mean, p, w, q, t, v, y

    def iterate(self, x=None, mean=None, q=None, v=None, args=None):
        """
        完全向量化加速后的采样与评估过程
        """
        # 1. 批量生成所有必需的随机数
        # 一次性生成 z_0 (等同于代码中的 z) 的基础各向同性高斯部分
        # z_0 = self._z_1 * N(0, I)
        z0 = self.rng_optimization.standard_normal((self._n_mirror_sampling, self.ndim_problem))

        # 2. 批量生成混合分量的系数和索引
        # 预先生成所有混合强度 l (self.ms) 对应的标准正态标量
        zk = self.rng_optimization.standard_normal((self._n_mirror_sampling, self.ms))

        # 3. 批量生成几何分布索引并映射到物理索引 (v)
        # v[(m - geometric(c_a) % m) - 1] 的逻辑向量化
        # 采样次数为 n_mirror_sampling * ms
        geom_samples = self.rng_optimization.geometric(self.c_a, size=(self._n_mirror_sampling, self.ms))
        idx_j = v[(self.m - geom_samples % self.m) - 1]

        # 4. 核心加速：利用矩阵乘法替代内部循环
        # 原逻辑：zq = sum_{j=1 to ms} z_k * q[idx_j]
        # 我们利用 q 的高级索引快速提取方向向量，其形状为 (n_mirror_sampling, ms, ndim_problem)
        q_selected = q[idx_j]

        # 利用 einsum 或 batch dot 计算加权和 zq
        # 'km,kmn->kn' 含义：k 是样本数，m 是混合强度，n 是维度
        # 这步替代了原代码中的 for _ in range(self.ms) 循环
        zq = np.einsum('km,kmn->kn', zk, q_selected)

        # 5. 合成最终的变异向量 z
        # z = sqrt(1-gamma)*z0 + sqrt(gamma/l)*zq
        z_final = self._z_1 * z0 + self._z_2 * zq

        # 6. 生成子代种群（包含镜像采样逻辑）
        x[:self._n_mirror_sampling] = mean + self.sigma * z_final

        # 处理镜像部分
        remaining = self.n_individuals - self._n_mirror_sampling
        if remaining > 0:
            x[self._n_mirror_sampling:self.n_individuals] = mean - self.sigma * z_final[:remaining]

        # Keep samples within search bounds to avoid runaway objective overflows.
        x = self._repair_bounds(x)

        # 7. 检查终止条件并批量计算适应度
        if self._check_terminations():
            return x, None  # 返回当前种群，y 将由 optimize 处理

        y = self._evaluate_fitness(x, args)  # 直接调用基类实现的批量评估
        y = self._sanitize_fitness_batch(y)

        return x, y

    def _update_distribution(self, x=None, mean=None, p=None, w=None, q=None,
                             t=None, v=None, y=None, y_bak=None):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        y_bak = np.asarray(y_bak, dtype=np.float64).reshape(-1)

        order = np.argsort(y)[:self.n_parents]
        y.sort()
        mean_w = np.dot(self._w[:self.n_parents], x[order])
        eps = 1e-12
        p = self._p_1*p + self._p_2*np.sqrt(self._mu_eff)*(mean_w - mean)/(self.sigma + eps)
        mean = mean_w
        if self._n_generations < self.m:
            q[self._n_generations] = p
        else:
            k_star = np.argmin(t[v[1:]] - t[v[:(self.m - 1)]])
            k_star += 1
            if t[v[k_star]] - t[v[k_star - 1]] > self.distance:
                k_star = 0
            v = np.append(np.append(v[:k_star], v[(k_star + 1):]), v[k_star])
            t[v[-1]], q[v[-1]] = self._n_generations, p
        # conduct success-based mutation strength adaptation
        l_w = np.dot(self._w, y_bak[:self.n_parents] > y[:self.n_parents])
        w = self._w_1*w + self._w_2*np.sqrt(self._mu_eff)*(2*l_w - 1)
        self.sigma *= np.exp(norm.cdf(w) - 1.0 + self.a_z)
        return mean, p, w, q, t, v

    def restart_reinitialize(self, args=None, x=None, mean=None, p=None, w=None, q=None,
                             t=None, v=None, y=None, fitness=None):
        if self.is_restart and ES.restart_reinitialize(self, y):
            x, mean, p, w, q, t, v, y = self.initialize(args, True)
            self._print_verbose_info(fitness, y[0])
        return x, mean, p, w, q, t, v, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, w, q, t, v, y = self.initialize(args)
        if np.asarray(mean).ndim != 1:
            raise ValueError(f"MMES expects mean to be 1D, got mean.shape={np.asarray(mean).shape}")
        self._print_verbose_info(fitness, y[0])
        while not self.termination_signal:
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            x, y = self.iterate(x, mean, q, v, args)
            if self._check_terminations():
                break
            if x.ndim != 2:
                raise ValueError(f"MMES expects x to be 2D, got x.shape={x.shape}")
            mean, p, w, q, t, v = self._update_distribution(x, mean, p, w, q, t, v, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(fitness, y)
            x, mean, p, w, q, t, v, y = self.restart_reinitialize(
                args, x, mean, p, w, q, t, v, y, fitness)
        results = self._collect(fitness, y, mean)
        results['p'] = p
        results['w'] = w
        return results
