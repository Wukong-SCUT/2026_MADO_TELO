from typing import Any, Callable, List, Optional

# 第14次修改：将gym库更新为gymnasium
# import gym  # 原代码已注释
import gym
import numpy as np

from env.parallel.worker import EnvWorker


class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.env = env_fn()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env, key, value)

    def reset(self) -> Any:
        return self.env.reset()

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        # Sequential EnvWorker objects are always ready
        return workers

    def send(self, action: Optional[np.ndarray]) -> None:
        if action is None:
            self.result = self.env.reset()
        else:
            self.result = self.env.step(action)

    def prepare_step(self, action: np.ndarray):
        return self.env.prepare_step(action)

    def prepare_generator_step(self, action: np.ndarray):
        self.send_prepare_generator_step(action)
        return self.recv_prepare_generator_step()

    def send_prepare_generator_step(self, action: np.ndarray) -> None:
        self.result = self.env.prepare_generator_step(action)

    def recv_prepare_generator_step(self):
        return self.result

    def prepare_actuator_step(self, action: np.ndarray):
        self.send_prepare_actuator_step(action)
        return self.recv_prepare_actuator_step()

    def send_prepare_actuator_step(self, action: np.ndarray) -> None:
        self.result = self.env.prepare_actuator_step(action)

    def recv_prepare_actuator_step(self):
        return self.result

    def commit_step(self, action: np.ndarray):
        self.send_commit_step(action)
        return self.recv_commit_step()

    def send_commit_step(self, action: np.ndarray) -> None:
        self.result = self.env.commit_step(action)

    def recv_commit_step(self):
        return self.result

    def seed(self, seed: Optional[int] = None) -> List[int]:
        super().seed(seed)
        return self.env.seed(seed)

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()
