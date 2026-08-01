from __future__ import division  # use // for integer division
from __future__ import absolute_import  # use from . import
from __future__ import print_function  # print function
from __future__ import unicode_literals  # all the strings are unicode

__author__ = 'Youhei Akimoto'

import os
import time
import warnings
from math import sqrt, exp, log, ceil, floor
import numpy as np
from numpy.random import randn
from numpy.linalg import norm, eigh

import torch

from . import utility
# print = utility.do_log

# Only suppress this known non-fatal tie warning from this module.
warnings.filterwarnings(
    "ignore",
    message="assumed no tie, but there exists",
    category=RuntimeWarning,
)

# import tensorflow as tf
# tf_log = 'tf_log'
# writer = tf.summary.create_file_writer(tf_log)

class VkdCma(object):
    """O(N*k^2 + k^3) Time/Space Variant of CMA-ES with C = D * (I + V * V^T) * D
    References
    ----------
    [1] Youhei Akimoto and Nikolaus Hansen.
    Online Model Selection for Restricted Covariance Matrix Adaptation.
    In Proc. of PPSN 2016, pp. 3--13 (2016)
    [2] Youhei Akimoto and Nikolaus Hansen.
    Projection-Based Restricted Covariance Matrix Adaptation for High
    Dimension. In Proc. of GECCO 2016, pp. 197--204 (2016)
    """

    def __init__(self, func, xmean0, sigma0, xs,  **kwargs):

        # ES Parameters
        self.N = len(xmean0)
        self.lam = kwargs.get('lam', int(4 + floor(3 * log(self.N))))
        wtemp = np.array([
            np.log(float(self.lam + 1) / 2.0) - np.log(1 + i)
            for i in range(self.lam // 2)
        ])
        self.w = kwargs.get('w', wtemp / np.sum(wtemp))
        self.sqrtw = np.sqrt(self.w)
        self.mueff = 1.0 / (self.w**2).sum()
        self.mu = self.w.shape[0]
        self.neval = 0

        # Arguments
        self.func = func
        self.xmean = np.array(xmean0)
        if isinstance(sigma0, np.ndarray):
            self.sigma = np.exp(np.log(sigma0).mean())
            self.D = sigma0 / self.sigma
        else:
            self.sigma = sigma0
            self.D = np.ones(self.N)

        # VkD Static Parameters
        self.k = kwargs.get('k_init', 0)  # alternatively, self.w.shape[0]
        self.kmin = kwargs.get('kmin', 0)
        self.kmax = kwargs.get('kmax', self.N - 1)
        assert (0 <= self.kmin <= self.kmax < self.N)
        self.k_inc_cond = kwargs.get('k_inc_cond', 30.0)
        self.k_dec_cond = kwargs.get('k_dec_cond', self.k_inc_cond)
        self.k_adapt_factor = kwargs.get('k_adapt_factor', 1.414)
        self.factor_sigma_slope = kwargs.get('factor_sigma_slope', 0.1)
        self.factor_diag_slope = kwargs.get(
            'factor_diag_slope', 1.0)  # 0.3 in PPSN (due to cc change)
        self.opt_conv = 0.5 * min(1, self.lam / self.N)
        self.accepted_slowdown = max(1., self.k_inc_cond / 10.)
        self.k_adapt_decay = 1.0 / self.N
        self.k_adapt_wait = 2.0 / self.k_adapt_decay - 1

        # VkD Dynamic Parameters
        self.k_active = 0
        self.last_log_sigma = np.log(self.sigma)
        self.last_log_d = 2.0 * np.log(self.D)
        self.last_log_cond_corr = np.zeros(self.N)
        self.ema_log_sigma = ExponentialMovingAverage(
            decay=self.opt_conv / self.accepted_slowdown, dim=1)
        self.ema_log_d = ExponentialMovingAverage(
            decay=self.k_adapt_decay, dim=self.N)
        self.ema_log_s = ExponentialMovingAverage(
            decay=self.k_adapt_decay, dim=self.N)
        self.itr_after_k_inc = 0

        # CMA Learning Rates
        self.cm = kwargs.get('cm', 1.0)
        (self.cone, self.cmu, self.cc) = self._get_learning_rate(self.k)

        # TPA Parameters
        self.cs = kwargs.get('cs', 0.3)
        self.ds = kwargs.get('ds', np.sqrt(self.N))  # or 4 - 3/N 
        self.flg_injection = False
        self.ps = 0

        # Initialize Dynamic Parameters
        self.V = np.zeros((self.k, self.N))
        self.S = np.zeros(self.N)
        self.pc = np.zeros(self.N)
        self.dx = np.zeros(self.N)
        self.U = np.zeros((self.N, self.k + self.mu + 1))
        self.arx = np.zeros((self.lam, self.N))
        self.arf = np.zeros(self.lam)

        # Stopping Condition
        self.fhist_len = 20 + self.N // self.lam
        self.tolf_checker = TolfChecker(self.fhist_len)
        self.ftarget = kwargs.get('ftarget', 1e-8)
        self.maxeval = kwargs.get('maxeval', 5e3 * self.N * self.lam)
        self.tolf = kwargs.get('tolf', abs(self.ftarget) / 1e5)
        self.tolfrel = kwargs.get('tolfrel', 1e-12)
        self.minstd = kwargs.get('minstd', 1e-12)
        self.minstdrel = kwargs.get('minstdrel', 1e-12)
        self.maxconds = kwargs.get('maxconds', 1e12)
        self.maxcondd = kwargs.get('maxcondd', 1e6)

        # Other Options
        self.batch_evaluation = kwargs.get('batch_evaluation', True)

        self.lb = kwargs.get('lb', -5 * np.ones(self.N))
        self.ub = kwargs.get('ub', 5 * np.ones(self.N))

        # Initialize with xs if provided
        xs = None
        if xs is not None:
            xs = np.copy(xs)

            # 修正维度 (应为 [n_samples, N])
            assert xs.ndim == 2 and xs.shape[1] == self.N, f"xs must be 2D with shape (_, {self.N})"

            n_given = xs.shape[0]

            if n_given > self.lam:
                # 如果多了，优先选择适应度最好的 lam 个
                xs = np.clip(xs, self.lb, self.ub)
                if self.batch_evaluation:
                    scores = self.func(torch.from_numpy(xs).cuda())
                else:
                    scores = np.array([self.func(x) for x in xs])
                self.neval += n_given
                idx = np.argsort(scores)[:self.lam]
                self.arx = xs[idx]
                self.arf = scores[idx]
            elif n_given < self.lam:
                # 如果少了，复制填充
                xs = np.clip(xs, self.lb, self.ub)
                reps = np.tile(xs, (int(np.ceil(self.lam / n_given)), 1))[:self.lam]
                self.arx = reps
                if self.batch_evaluation:
                    self.arf = self.func(torch.from_numpy(self.arx).cuda())
                else:
                    self.arf = np.array([self.func(x) for x in self.arx])
                self.neval += self.lam
            else:
                self.arx = np.clip(xs, self.lb, self.ub)
                if self.batch_evaluation:
                    self.arf = self.func(torch.from_numpy(self.arx).cuda())
                else:
                    self.arf = np.array([self.func(x) for x in self.arx])
                self.neval += self.lam

            self.xmean = np.mean(self.arx, axis=0)
            self.dx = self.xmean - np.array(xmean0)
        else:
            self.arx = np.zeros((self.lam, self.N))
            self.arf = np.zeros(self.lam)

    # === 在 VkdCma 类中新增 ===

    def incorporate_generation(self, arx, arf):
        """
        将一代已经采样并评估好的 (arx, arf) 注入并执行更新（不做额外评估）。
        arx: ndarray shape (lam, N)
        arf: ndarray shape (lam,)
        注意：本函数**不负责**增加 neval（由调用者负责递增 neval），
        但会执行与 _onestep() 中评估后完全相同的参数更新逻辑。
        """
        arx = np.asarray(arx)
        arf = np.asarray(arf)

        # place the population & fitness
        self.arx = arx.copy()
        self.arf = arf.copy()

        # normalized steps (same definition as in _onestep sampling part)
        ary = (self.arx - self.xmean) / self.sigma

        # sorting & selection (same as in _onestep)
        idx = np.argsort(self.arf)
        if not np.all(self.arf[idx[1:]] - self.arf[idx[:-1]] > 0.):
            warnings.warn("assumed no tie, but there exists", RuntimeWarning)

        sary = ary[idx[:self.mu]]

        # ------ Update xmean ------
        self.dx = np.dot(self.w, sary)
        self.dz = self._inv_sqrt_ivv(self.dx / self.D)  # For flg_dev_kadapt
        self.xmean += (self.cm * self.sigma) * self.dx

        # ------ TPA / injection handling ------
        if self.flg_injection:
            # compute relative rank difference of injected pair (same as original)
            alpha_act = np.where(idx == 1)[0][0] - np.where(idx == 0)[0][0]
            alpha_act /= float(self.lam - 1)
            self.ps += self.cs * (alpha_act - self.ps)
            self.sigma *= np.exp(self.ps / self.ds)
            hsig = self.ps < 0.5
        else:
            # in the original code, flg_injection is set True on first step
            self.flg_injection = True
            hsig = True

        # ------ Cumulation for covariance ------
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * self.dx

        # ------ Update V, S and D (covariance structure) ------
        k = self.k
        ka = self.k_active

        if self.cmu == 0.0:
            rankU = ka + 1
            alpha = np.sqrt(
                abs(1 - self.cmu - self.cone + self.cone * (1 - hsig) * self.cc * (2 - self.cc)))
            # U uses V, S
            if ka > 0:
                self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, rankU - 1] = np.sqrt(self.cone) * (self.pc / self.D)
        elif self.cone == 0.0:
            rankU = ka + self.mu
            alpha = np.sqrt(
                abs(1 - self.cmu - self.cone + self.cone * (1 - hsig) * self.cc * (2 - self.cc)))
            if ka > 0:
                self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, ka:rankU] = np.sqrt(self.cmu) * self.sqrtw * (sary / self.D).T
        else:
            rankU = ka + self.mu + 1
            alpha = np.sqrt(
                abs(1 - self.cmu - self.cone + self.cone * (1 - hsig) * self.cc * (2 - self.cc)))
            if ka > 0:
                self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, ka:rankU - 1] = np.sqrt(self.cmu) * self.sqrtw * (sary / self.D).T
            self.U[:, rankU - 1] = np.sqrt(self.cone) * (self.pc / self.D)

        if self.N > rankU:
            # O(N k^2 + k^3)
            DD, R = np.linalg.eigh(np.dot(self.U[:, :rankU].T, self.U[:, :rankU]))
            idxeig = np.argsort(DD)[::-1]
            gamma = 0 if rankU <= k else DD[idxeig[k:]].sum() / (self.N - k)
            beta = alpha * alpha + gamma

            self.k_active = ka = min(np.sum(DD >= 0), k)
            if ka > 0:
                self.S[:ka] = (DD[idxeig[:ka]] - gamma) / beta
                self.V[:ka] = (np.dot(self.U[:, :rankU], R[:, idxeig[:ka]]) / np.sqrt(DD[idxeig[:ka]])).T
        else:
            DD, L = np.linalg.eigh(np.dot(self.U[:, :rankU], self.U[:, :rankU].T))
            idxeig = np.argsort(DD)[::-1]
            gamma = 0 if rankU <= k else DD[idxeig[k:]].sum() / (self.N - k)
            beta = alpha * alpha + gamma

            self.k_active = ka = min(np.sum(DD >= 0), k)
            if ka > 0:
                self.S[:ka] = (DD[idxeig[:ka]] - gamma) / beta
                self.V[:ka] = L[:, idxeig[:ka]].T

        # update D (scale) using U etc.
        denom = (1.0 + np.dot(self.S[:ka], self.V[:ka] * self.V[:ka])) if ka > 0 else 1.0
        self.D *= np.sqrt(
            (alpha * alpha + np.sum(self.U[:, :rankU] * self.U[:, :rankU], axis=1)) / denom)

        # Covariance normalization by determinant
        gmean_eig = np.exp(self._get_log_determinant_of_cov() / self.N / 2.0)
        self.D /= gmean_eig
        self.pc /= gmean_eig

        # ------ k-adaptation bookkeeping ------
        self.itr_after_k_inc += 1

        # EMA updates
        self.ema_log_sigma.update(np.log(self.sigma) - self.last_log_sigma)
        self.lnsigma_change = self.ema_log_sigma.M / (self.opt_conv / self.accepted_slowdown)
        self.last_log_sigma = np.log(self.sigma)

        self.ema_log_d.update(
            2. * np.log(self.D) + np.log(1 + np.dot(self.S[:self.k], self.V[:self.k] ** 2)) - self.last_log_d)
        self.lndiag_change = self.ema_log_d.M / (self.cmu + self.cone)
        self.last_log_d = 2. * np.log(self.D) + np.log(1 + np.dot(self.S[:self.k], self.V[:self.k] ** 2))

        self.ema_log_s.update(np.log(1 + self.S) - self.last_log_cond_corr)
        self.lnlambda_change = self.ema_log_s.M / (self.cmu + self.cone)
        self.last_log_cond_corr = np.log(1 + self.S)

        # k increase/decrease logic (same as original)
        flg_k_increase = self.itr_after_k_inc > self.k_adapt_wait
        flg_k_increase *= self.k < self.kmax
        if self.k > 0:
            flg_k_increase *= np.all((1 + self.S[:self.k]) > self.k_inc_cond)
        flg_k_increase *= (np.abs(self.lnsigma_change) < self.factor_sigma_slope)
        flg_k_increase *= np.all(np.abs(self.lndiag_change) < self.factor_diag_slope)

        flg_k_decrease = (self.k > self.kmin) * (1 + self.S[:self.k] < self.k_dec_cond)
        flg_k_decrease *= (self.lnlambda_change[:self.k] < 0.) if self.k > 0 else False

        if (self.itr_after_k_inc > self.k_adapt_wait) and flg_k_increase:
            # increase k
            self.k_active = k
            newk = min(max(int(np.ceil(self.k * self.k_adapt_factor)), self.k + 1), self.kmax)
            self.V = np.vstack((self.V, np.zeros((newk - k, self.N))))
            self.U = np.empty((self.N, newk + self.mu + 1))
            (self.cone, self.cmu, self.cc) = self._get_learning_rate(self.k)
            self.itr_after_k_inc = 0
        elif self.itr_after_k_inc > k * self.k_adapt_wait and np.any(flg_k_decrease):
            flg_keep = np.logical_not(flg_k_decrease)
            new_k = max(np.count_nonzero(flg_keep), self.kmin)
            self.V = self.V[flg_keep]
            self.S[:new_k] = (self.S[:flg_keep.shape[0]])[flg_keep]
            self.S[new_k:] = 0
            self.k = self.k_active = new_k
            (self.cone, self.cmu, self.cc) = self._get_learning_rate(self.k)

        # final covariance normalization (same as original)
        gmean_eig = np.exp(self._get_log_determinant_of_cov() / self.N / 2.0)
        self.D /= gmean_eig
        self.pc /= gmean_eig

    def replay_history(self, history):
        """
        history: iterable of {"x": ndarray, "y": ndarray}
        For each saved generation, incorporate it and advance neval & tolf checker.
        """
        for record in history:
            arx = np.asarray(record['x'])
            arf = np.asarray(record['y'])

            # incorporate generation update (does NOT touch neval)
            self.incorporate_generation(arx, arf)

            # increase evaluation count to reflect the replayed evaluations
            self.neval += arx.shape[0] if (arx.ndim == 2) else len(arf)

            # update tolf history as if _check() had been called after that generation
            try:
                self.tolf_checker.update(self.arf)
            except Exception:
                # defensive: don't fail replay if tolf_checker misbehaves
                pass

    def run(self):

        itr = 0
        satisfied = False
        while not satisfied:
            itr += 1
            self._onestep()
            satisfied, condition = self._check()
            if itr % 20 == 0:
                print(itr, self.neval, self.arf.min(), self.sigma)
            if satisfied:
                print(condition)
        return self.xmean

    def _repair_bounds(self, x):
        """Clip individuals to remain within [lb, ub] bounds."""
        return np.clip(x, self.lb, self.ub)

    def _onestep(self):

        # ======================================================================
        # VkD-CMA (GECCO 2016)

        k = self.k
        ka = self.k_active

        # Sampling
        if True:
            # Sampling with two normal vectors
            # Available only if S >= 0
            arzd = randn(self.lam, self.N)
            arzv = randn(self.lam, ka)
            ary = (arzd + np.dot(arzv * np.sqrt(self.S[:ka]), self.V[:ka])
                   ) * self.D
        else:
            # Sampling with one normal vectors
            # Available even if S < 0 as long as V are orthogonal to each other
            arz = randn(self.lam, self.N)
            ary = arz + np.dot(
                np.dot(arz, self.V[:ka].T) *
                (np.sqrt(1.0 + self.S[:ka]) - 1.0), self.V[:ka])
            ary *= self.D

        # Injection
        if self.flg_injection:
            mnorm = self._mahalanobis_square_norm(self.dx)
            dy = (norm(randn(self.N)) / sqrt(mnorm)) * self.dx
            ary[0] = dy
            ary[1] = -dy
        self.arx = self.xmean + self.sigma * ary

        self.arx = self._repair_bounds(self.arx)

        # self.arx = np.clip(self.arx, -5, 5)
        # print(f"{np.mean(self.arx):.2e} | {self.sigma:.2e}", end='\r')

        # Evaluation
        if self.batch_evaluation:
            # self.arf = self.func(self.arx)
            # self.neval += self.lam

            arx_tensor = torch.from_numpy(self.arx).cuda()  # 转换为GPU张量
            arf_tensor = self.func(arx_tensor)  # 假设func已适配GPU
            self.arf = arf_tensor#.cpu().numpy()
            self.neval += self.lam

        else:
            self.arf = np.zeros(self.lam)
            for i in range(self.lam):
                self.arf[i] = self.func(self.arx[i])
                self.neval += 1



        idx = np.argsort(self.arf)
        if not np.all(self.arf[idx[1:]] - self.arf[idx[:-1]] > 0.):
            warnings.warn("assumed no tie, but there exists", RuntimeWarning)

        sary = ary[idx[:self.mu]]

        # Update xmean
        self.dx = np.dot(self.w, sary)
        self.dz = self._inv_sqrt_ivv(self.dx / self.D)  # For flg_dev_kadapt
        self.xmean += (self.cm * self.sigma) * self.dx

        # TPA (PPSN 2014 version)
        if self.flg_injection:
            alpha_act = np.where(idx == 1)[0][0] - np.where(idx == 0)[0][0]
            alpha_act /= float(self.lam - 1)
            self.ps += self.cs * (alpha_act - self.ps)
            # assert abs(self.sigma) < 1e5
            # print(str(self.sigma) ,end= '\r')
            self.sigma *= exp(self.ps / self.ds)
            hsig = self.ps < 0.5
        else:
            self.flg_injection = True
            hsig = True


        # with writer.as_default():
        #     tf.summary.scalar('sigma', self.sigma, step=itr)
        #     tf.summary.scalar('arx_mean', np.mean(self.arx), step=itr)
        #     tf.summary.scalar('ps', self.ps, step=itr)
        #     tf.summary.scalar('ds', self.ds, step=itr)
        #     tf.summary.scalar('cs', self.cs, step=itr)

        # Cumulation
        self.pc = (1 - self.cc) * self.pc + hsig * sqrt(self.cc * (2 - self.cc)
                                                        * self.mueff) * self.dx

        # Update V, S and D
        # Cov = D(alpha**2 * I + UU^t)D
        if self.cmu == 0.0:
            rankU = ka + 1
            alpha = sqrt(
                abs(1 - self.cmu - self.cone + self.cone * (1 - hsig) * self.cc
                    * (2 - self.cc)))
            self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, rankU - 1] = sqrt(self.cone) * (self.pc / self.D)
        elif self.cone == 0.0:
            rankU = ka + self.mu
            alpha = sqrt(
                abs(1 - self.cmu - self.cone + self.cone * (1 - hsig) * self.cc
                    * (2 - self.cc)))
            self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, ka:rankU] = sqrt(self.cmu) * self.sqrtw * (sary /
                                                                 self.D).T
        else:
            rankU = ka + self.mu + 1
            alpha = sqrt(
                abs(1 - self.cmu - self.cone + self.cone * (1 - hsig) * self.cc
                    * (2 - self.cc)))
            self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, ka:rankU - 1] = sqrt(self.cmu) * self.sqrtw * (sary /
                                                                     self.D).T
            self.U[:, rankU - 1] = sqrt(self.cone) * (self.pc / self.D)

        if self.N > rankU:
            # O(Nk^2 + k^3)
            DD, R = eigh(np.dot(self.U[:, :rankU].T, self.U[:, :rankU]))
            idxeig = np.argsort(DD)[::-1]
            gamma = 0 if rankU <= k else DD[idxeig[k:]].sum() / (self.N - k)
            beta = alpha * alpha + gamma

            self.k_active = ka = min(np.sum(DD >= 0), k)
            self.S[:ka] = (DD[idxeig[:ka]] - gamma) / beta
            self.V[:ka] = (np.dot(self.U[:, :rankU], R[:, idxeig[:ka]]) /
                           np.sqrt(DD[idxeig[:ka]])).T
        else:
            # O(N^3 + N^2(k+mu+1))
            # If this is the case, the standard CMA is preferred
            DD, L = eigh(np.dot(self.U[:, :rankU], self.U[:, :rankU].T))
            idxeig = np.argsort(DD)[::-1]
            gamma = 0 if rankU <= k else DD[idxeig[k:]].sum() / (self.N - k)
            beta = alpha * alpha + gamma

            self.k_active = ka = min(np.sum(DD >= 0), k)
            self.S[:ka] = (DD[idxeig[:ka]] - gamma) / beta
            self.V[:ka] = L[:, idxeig[:ka]].T

        self.D *= np.sqrt(
            (alpha * alpha + np.sum(
                self.U[:, :rankU] * self.U[:, :rankU], axis=1)) /
            (1.0 + np.dot(self.S[:ka], self.V[:ka] * self.V[:ka])))

        # Covariance Normalization by Its Determinant
        gmean_eig = np.exp(self._get_log_determinant_of_cov() / self.N / 2.0)
        self.D /= gmean_eig
        self.pc /= gmean_eig

        # ======================================================================
        # k-Adaptation (PPSN 2016)
        self.itr_after_k_inc += 1

        # Exponential Moving Average
        self.ema_log_sigma.update(log(self.sigma) - self.last_log_sigma)
        self.lnsigma_change = self.ema_log_sigma.M / (self.opt_conv /
                                                      self.accepted_slowdown)
        self.last_log_sigma = log(self.sigma)
        self.ema_log_d.update(2. * np.log(self.D) + np.log(1 + np.dot(
            self.S[:self.k], self.V[:self.k]**2)) - self.last_log_d)
        self.lndiag_change = self.ema_log_d.M / (self.cmu + self.cone)
        self.last_log_d = 2. * np.log(
            self.D) + np.log(1 + np.dot(self.S[:self.k], self.V[:self.k]**2))
        self.ema_log_s.update(np.log(1 + self.S) - self.last_log_cond_corr)
        self.lnlambda_change = self.ema_log_s.M / (self.cmu + self.cone)
        self.last_log_cond_corr = np.log(1 + self.S)

        # Check for adaptation condition
        flg_k_increase = self.itr_after_k_inc > self.k_adapt_wait
        flg_k_increase *= self.k < self.kmax
        flg_k_increase *= np.all((1 + self.S[:self.k]) > self.k_inc_cond)
        flg_k_increase *= (
            np.abs(self.lnsigma_change) < self.factor_sigma_slope)
        flg_k_increase *= np.all(
            np.abs(self.lndiag_change) < self.factor_diag_slope)
        # print(self.itr_after_k_inc > self.k_adapt_wait, self.k < self.kmax,
        #       np.all((1 + self.S[:self.k]) > self.k_inc_cond),
        #       np.abs(self.lnsigma_change) < self.factor_sigma_slope,
        #       np.percentile(np.abs(self.lndiag_change), [1, 50, 99]))

        flg_k_decrease = (self.k > self.kmin) * (
            1 + self.S[:self.k] < self.k_dec_cond)
        flg_k_decrease *= (self.lnlambda_change[:self.k] < 0.)

        if (self.itr_after_k_inc > self.k_adapt_wait) and flg_k_increase:
            # ----- Increasing k -----
            self.k_active = k
            self.k = newk = min(
                max(int(ceil(self.k * self.k_adapt_factor)), self.k + 1),
                self.kmax)
            self.V = np.vstack((self.V, np.zeros((newk - k, self.N))))
            self.U = np.empty((self.N, newk + self.mu + 1))
            # update constants
            (self.cone, self.cmu, self.cc) = self._get_learning_rate(self.k)
            self.itr_after_k_inc = 0

        elif self.itr_after_k_inc > k * self.k_adapt_wait and np.any(
                flg_k_decrease):
            # ----- Decreasing k -----
            flg_keep = np.logical_not(flg_k_decrease)
            new_k = max(np.count_nonzero(flg_keep), self.kmin)
            self.V = self.V[flg_keep]
            self.S[:new_k] = (self.S[:flg_keep.shape[0]])[flg_keep]
            self.S[new_k:] = 0
            self.k = self.k_active = new_k
            # update constants
            (self.cone, self.cmu, self.cc) = self._get_learning_rate(self.k)
        # ==============================================================================

        # Covariance Normalization by Its Determinant
        gmean_eig = exp(self._get_log_determinant_of_cov() / self.N / 2.0)
        self.D /= gmean_eig
        self.pc /= gmean_eig

    def _mahalanobis_square_norm(self, dx):
        """Square norm of dx w.r.t. C = D*(I + V*S*V^t)*D
        Parameters
        ----------
        dx : numpy.ndarray (1D)
        Returns
        -------
        square of the Mahalanobis distance dx^t * (D*(I + V*S*V^t)*D)^{-1} * dx
        """
        D = self.D
        V = self.V[:self.k_active]
        S = self.S[:self.k_active]

        dy = dx / D
        vdy = np.dot(V, dy)
        return np.sum(dy * dy) - np.sum((vdy * vdy) * (S / (S + 1.0)))

    def _inv_sqrt_ivv(self, vec):
        """Return (I + V*V^t)^{-1/2} x"""
        if self.k_active == 0:
            return vec
        else:
            return vec + np.dot(
                np.dot(self.V[:self.k_active], vec) *
                (1.0 / np.sqrt(1.0 + self.S[:self.k_active]) - 1.0
                 ), self.V[:self.k_active])

    def _get_learning_rate(self, k):
        """Return the learning rate cone, cmu, cc depending on k
        Parameters
        ----------
        k : int
            the number of vectors for covariance matrix
        Returns
        -------
        cone, cmu, cc : float in [0, 1]. Learning rates for rank-one, rank-mu,
         and the cumulation factor for rank-one.
        """
        nelem = self.N * (k + 1)
        cone = 2.0 / (nelem + self.N + 2 * (k + 2) + self.mueff)  # PPSN 2016
        # cone = 2.0 / (nelem + 2 * (k + 2) + self.mueff)  # GECCO 2016
        # cc = (4 + self.mueff / self.N) / (
        #     (self.N + 2 * (k + 1)) / 3 + 4 + 2 * self.mueff / self.N)

        # New Cc and C1: Best Cc depends on C1, not directory on K.
        # Observations on Cigar (N = 3, 10, 30, 100, 300, 1000) by Rank-1 VkD.
        cc = sqrt(cone)
        cmu = min(1 - cone, 2.0 * (self.mueff - 2 + 1.0 / self.mueff) /
                  (nelem + 4 * (k + 2) + self.mueff))
        return cone, cmu, cc

    def _get_log_determinant_of_cov(self):
        return 2.0 * np.sum(np.log(self.D)) + np.sum(
            np.log(1.0 + self.S[:self.k_active]))

    def _check(self):
        is_satisfied = False
        condition = ''
        self.tolf_checker.update(self.arf)
        std = self.sigma * exp(self._get_log_determinant_of_cov() / self.N /
                               2.0)

        if self.arf.min() <= self.ftarget:
            is_satisfied = True
            condition = 'ftarget'
        if not is_satisfied and self.neval >= self.maxeval:
            is_satisfied = True
            condition = 'maxeval'
        if not is_satisfied and self.tolf_checker.check_relative(self.tolfrel):
            is_satisfied = True
            condition = 'tolfrel'
        if not is_satisfied and self.tolf_checker.check_absolute(self.tolf):
            is_satisfied = True
            condition = 'tolf'
        if not is_satisfied and self.tolf_checker.check_flatarea():
            is_satisfied = True
            condition = 'flatarea'
        if not is_satisfied and std < self.minstd:
            is_satisfied = True
            condition = 'minstd'
        if not is_satisfied and std < self.minstd * np.median(
                np.abs(self.xmean)):
            is_satisfied = True
            condition = 'minstdrel'
        if not is_satisfied and np.any(
                1 + self.S[:self.k_active] > self.maxconds):
            is_satisfied = True
            condition = 'maxconds'
        if not is_satisfied and self.D.max() / self.D.min() > self.maxcondd:
            is_satisfied = True
            condition = 'maxcondd'
        return is_satisfied, condition


class ExponentialMovingAverage(object):
    """Exponential Moving Average, Variance, and SNR (Signal-to-Noise Ratio)
    See http://www-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
    """

    def __init__(self, decay, dim, flg_init_with_data=False):
        """
        The latest N steps occupy approximately 86% of the information when
        decay = 2 / (N - 1).
        """
        self.decay = decay
        self.M = np.zeros(dim)  # Mean Estimate
        self.S = np.zeros(dim)  # Variance Estimate
        self.flg_init = -flg_init_with_data

    def update(self, datum):
        a = self.decay if self.flg_init else 1.
        self.S += a * ((1 - a) * (datum - self.M)**2 - self.S)
        self.M += a * (datum - self.M)


class TolfChecker(object):
    def __init__(self, size=20):
        """
        Parameters
        ----------
        size : int
            number of points for which the value is restored
        """
        self._min_hist = np.empty(size) * np.nan
        self._l_quartile_hist = np.empty(size) * np.nan
        # self._median_hist = np.empty(size) * np.nan
        self._u_quartile_hist = np.empty(size) * np.nan
        # self._max_hist = np.empty(size) * np.nan
        # self._pop_hist = np.empty(size) * np.nan
        self._next_position = 0

    def update(self, arf):
        self._min_hist[self._next_position] = np.nanmin(arf)
        self._l_quartile_hist[self._next_position] = np.nanpercentile(arf, 25)
        # self._median_hist[self._next_position] = np.nanmedian(arf)
        self._u_quartile_hist[self._next_position] = np.nanpercentile(arf, 75)
        # self._max_hist[self._next_position] = np.nanmax(arf)
        self._next_position = (
            self._next_position + 1) % self._min_hist.shape[0]

    def check(self, tolfun=1e-9):
        # alias to check_absolute
        return self.check_relative(tolfun)

    def check_relative(self, tolfun=1e-9):
        iqr = np.nanmedian(self._u_quartile_hist - self._l_quartile_hist)
        return iqr < tolfun * np.abs(np.nanmedian(self._min_hist))

    def check_absolute(self, tolfun=1e-9):
        iqr = np.nanmedian(self._u_quartile_hist - self._l_quartile_hist)
        return iqr < tolfun

    def check_flatarea(self):
        return np.nanmedian(self._l_quartile_hist - self._min_hist) == 0


def default_option(dimension, FEs, lb=-5, ub=5):
    N = dimension
    esoption = dict()
    esoption['lam'] = int(4 + 3 * log(N))
    esoption['ds'] = 4 - 3 / N  # sqrt(N) in PPSN
    # Termination Condition
    tcoption = dict()
    tcoption['ftarget'] = 1e-20

    tcoption['maxeval'] = FEs  # int(5e3 * N * esoption['lam'])
    # tcoption['batch_evaluation'] = (True if torch.cuda.is_available() else False)
    tcoption['batch_evaluation'] = False

    # tcoption['lam'] = 100

    # utility.do_log(f"maxeval: {tcoption['maxeval']}, batch_evaluation: {tcoption['batch_evaluation']}, lam: {tcoption['lam']}")
    # utility.do_log(f"maxeval: {tcoption['maxeval']}, batch_evaluation: {tcoption['batch_evaluation']}")

    tcoption['tolf'] = 1e-12  # 1e-12
    tcoption['tolfrel'] = 1e-12  # 1e-12
    tcoption['minstd'] = 1e-12
    tcoption['minstdrel'] = 1e-12
    tcoption['maxconds'] = 1e12
    tcoption['maxcondd'] = 1e6
    # k-adaptation
    kaoption = dict()
    kaoption['kmin'] = 0
    kaoption['kmax'] = N - 1
    kaoption['k_init'] = kaoption['kmin']
    kaoption['k_inc_cond'] = 30.0
    kaoption['k_dec_cond'] = kaoption['k_inc_cond']
    kaoption['k_adapt_factor'] = 1.414
    kaoption['factor_sigma_slope'] = 0.1
    kaoption['factor_diag_slope'] = 1.0  # 0.3 in PPSN

    poption = dict()
    poption['lb'] = lb
    poption['ub'] = ub

    opts = {**esoption, **tcoption, **kaoption, **poption}
    # opts.update(esoption)
    # opts.update(tcoption)
    # opts.update(kaoption)
    return opts




if __name__ == '__main__':

    from cec2013lsgo_torch.cec2013 import Benchmark
    import time
    import torch
    import numpy as np

    bench = Benchmark(device='cuda' if torch.cuda.is_available() else 'cpu', output_format='numpy')
    assert torch.cuda.is_available()
    # 初始化全局列表存储所有问题的统计结果
    init_fitness_avg_list = []
    init_fitness_std_list = []
    fina_fitness_avg_list = []
    fina_fitness_std_list = []
    time_list = []

    for i in range(1, 16):
        problem_start_time = time.time()
        utility.do_log('=' * 10 + f"problem {i}")

        # 当前问题的数据收集列表
        init_fitness_list = []
        fina_fitness_list = []
        for j in range(1):  # 运行5次独立试验
            trial_start_time = time.time()
            seed = j
            np.random.seed(seed)

            fun = bench.get_function(i)
            info = bench.get_info(i)

            N = info['dimension']
            fobj = fun
            xmean0 = 3. + 2. * randn(N)
            sigma0 = 2.

            # Optional Parameters
            esoption = dict()
            esoption['lam'] = int(4 + 3 * log(N))
            esoption['ds'] = 4 - 3 / N  # sqrt(N) in PPSN
            # Termination Condition
            tcoption = dict()
            tcoption['ftarget'] = 1e-20

            tcoption['maxeval'] = 3E6 // 1 # int(5e3 * N * esoption['lam'])
            tcoption['batch_evaluation'] = (True if torch.cuda.is_available() else False)
            # tcoption['lam'] = 100

            # utility.do_log(f"maxeval: {tcoption['maxeval']}, batch_evaluation: {tcoption['batch_evaluation']}, lam: {tcoption['lam']}")
            utility.do_log(f"maxeval: {tcoption['maxeval']}, batch_evaluation: {tcoption['batch_evaluation']}")

            tcoption['tolf'] = 1e-12 # 1e-12
            tcoption['tolfrel'] = 1e-12 # 1e-12
            tcoption['minstd'] = 1e-12
            tcoption['minstdrel'] = 1e-12
            tcoption['maxconds'] = 1e12
            tcoption['maxcondd'] = 1e6
            # k-adaptation
            kaoption = dict()
            kaoption['kmin'] = 0
            kaoption['kmax'] = N - 1
            kaoption['k_init'] = kaoption['kmin']
            kaoption['k_inc_cond'] = 30.0
            kaoption['k_dec_cond'] = kaoption['k_inc_cond']
            kaoption['k_adapt_factor'] = 1.414
            kaoption['factor_sigma_slope'] = 0.1
            kaoption['factor_diag_slope'] = 1.0  # 0.3 in PPSN

            opts = dict()
            opts.update(esoption)
            opts.update(tcoption)
            opts.update(kaoption)

            # 运行优化过程
            init_fitness = None
            fina_fitness = None
            itr = 0
            for r in range(10):  # 10次重启
                vkd = VkdCma(fun, xmean0, sigma0, **opts)
                satisfied = False
                while not satisfied:
                    if itr == 1:  # 记录初始适应度
                        init_fitness = vkd.arf.min()
                    itr += 1
                    vkd._onestep()
                    satisfied, condition = vkd._check()
                if condition == 'maxeval':
                    break
                opts['lam'] = min(opts['lam'] * 2, 100)

            fina_fitness = vkd.arf.min()

            # 收集数据
            if init_fitness is not None:
                init_fitness_list.append(init_fitness)
            fina_fitness_list.append(fina_fitness)

            utility.do_log(f"Trial {j + 1} time: {time.time() - trial_start_time:.2f}s")

            # 计算统计量
        init_avg = np.mean(init_fitness_list)
        init_std = np.std(init_fitness_list)
        fina_avg = np.mean(fina_fitness_list)
        fina_std = np.std(fina_fitness_list)
        total_time = time.time() - problem_start_time

        # 存储结果
        init_fitness_avg_list.append(f"{init_avg:.4e}")
        init_fitness_std_list.append(f"{init_std:.4e}")
        fina_fitness_avg_list.append(f"{fina_avg:.4e}")
        fina_fitness_std_list.append(f"{fina_std:.4e}")
        time_list.append(f"{total_time:.2f}s")

        # 输出当前问题的统计结果
        utility.do_log(
            f"VkdCMA: problem {i}\n"
            f"Initial fitness: {init_avg:.2e} ± {init_std:.2e}\n"
            f"Final fitness: {fina_avg:.2e} ± {fina_std:.2e}\n"
            f"Total time: {total_time:.2f}s\n"
            + '=' * 50
        )

    # 输出最终汇总结果
    utility.do_log("\n\nFinal Summary:")
    utility.do_log(f"Initial Fitness (avg): {init_fitness_avg_list}")
    utility.do_log(f"Initial Fitness (std): {init_fitness_std_list}")
    utility.do_log(f"Final Fitness (avg): {fina_fitness_avg_list}")
    utility.do_log(f"Final Fitness (std): {fina_fitness_std_list}")
    utility.do_log(f"Problem Times: {time_list}")
    
# writer.close()
