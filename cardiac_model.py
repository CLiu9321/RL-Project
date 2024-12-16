import numpy as np
import tensorflow.compat.v1 as tf
import gym
from gym import spaces

def state_range(rat_type):
    if rat_type == 'healthy_stable':
        print('[cardiac_model/healthy_stable]')
        return -0.49, 0.51, -0.36, 0.64
    elif rat_type == 'healthy_exercise':
        print('[cardiac_model/healthy_exercise]')
        return -0.64, 0.36, -0.62, 0.38
    elif rat_type == 'hypertension_stable':
        print('[cardiac_model/hypertension_stable]')
        return -0.62, 0.38, -0.67, 0.33
    elif rat_type == 'hypertension_exercise':
        print('[cardiac_model/hypertension_exercise]')
        return -0.62, 0.38, -0.67, 0.33
    else:
        raise ValueError(f"Unknown rat_type: {rat_type}")

class CardiacModel_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, tcn_model, rat_type, noise_level=0.0):
        super(CardiacModel_Env, self).__init__()
        print(tf.__version__)

        self.tcn_model = tcn_model
        self.min_HR, self.max_HR, self.min_MAP, self.max_MAP = state_range(rat_type)
        self.noise_level = noise_level

        # 6维动作空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        # 状态空间 (HR, MAP)
        low = np.array([self.min_HR, self.min_MAP], dtype=np.float32)
        high = np.array([self.max_HR, self.max_MAP], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.sp_list = [
            np.array([0.074, -0.37], dtype=np.float32),
            np.array([-0.355, 0.108], dtype=np.float32),
            np.array([0.064, -0.16], dtype=np.float32)
        ]
        self.setpoints = self.sp_list[2]
        self.initial_state = np.array([0.3, 0.25], dtype=np.float32)

        self.ep_length = 200
        self.current_step = 0
        self.num_episode = 0
        self.state = self.initial_state.copy()
        self.previous_action = np.zeros((1, 6), dtype=np.float32)

        # 你可以自己指定 sigma_c，决定reward曲线陡峭度
        self.sigma_c = 0.5

    def step(self, action):
        action = np.array(action, dtype=np.float32).reshape(-1,)
        action = action / 2.0 + 0.28   # 保留你的动作缩放

        # TCN预测
        in_TCN = np.concatenate((action, self.state), axis=0).reshape(1, 1, 8)
        predicted = self.tcn_model.predict(in_TCN)[0][0]

        # 加噪声
        noise = np.random.normal(0, self.noise_level * 0.5, size=self.state.shape)
        self.state = predicted + noise

        # 引导状态朝目标靠近
        self.state += 0.1 * (self.setpoints - self.state)

        # 可选：在 step=100 时切换setpoint
        if self.current_step == 100:
            self.setpoints = self.sp_list[0]

        reward = self.reward_fn(self.state, self.setpoints)
        self.current_step += 1
        terminated = (self.current_step >= self.ep_length)
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def reward_fn(self, state, setpoint):
        distance_sq = np.sum((state - setpoint)**2)
        # c(x) = 1 - exp(||x - x_target||^2 / sigma^2)
        reward = 1.0 - np.exp(distance_sq / (self.sigma_c**2))
        return reward

    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed)
        self.num_episode += 1
        self.current_step = 0
        self.state = self.initial_state.copy()
        self.previous_action = np.zeros((1,6), dtype=np.float32)

        return self.state, {}

    def render(self, mode='human'):
        pass