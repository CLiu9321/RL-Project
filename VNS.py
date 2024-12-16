import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 使用上面更新过的 CardiacModel_Env (ep_length=200, step=100换setpoint)
from cardiac_model import CardiacModel_Env
from tcn_model import TCN_config

def load_tcn_model(rat_type):
    return TCN_config(rat_type=rat_type)

def make_env(rat_type, noise_level):
    def _init():
        tcn_model = load_tcn_model(rat_type)
        env = CardiacModel_Env(tcn_model, rat_type=rat_type, noise_level=noise_level)
        return env
    return _init

def initialize_environment(rat_type='healthy_stable', noise_level=0.1):
    env = DummyVecEnv([make_env(rat_type, noise_level)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    return env

def train_td3(env, total_timesteps=500, learning_rate=1e-5, buffer_size=50000, log_dir="./logs"):
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
    policy_kwargs = dict(net_arch=[64, 64])

    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=128,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir
    )

    new_logger = configure(log_dir, ["stdout","csv"])
    model.set_logger(new_logger)

    print(f"Starting TD3 training for {log_dir}...")
    model.learn(total_timesteps=total_timesteps)
    print("TD3 training completed.")
    return model

def save_model(model, filename="td3_cardiac_model"):
    model.save(filename)
    print(f"Model saved as {filename}.")


def evaluate_model(model, env, num_steps=200, env_label=""):

    obs = env.reset()
    total_reward = 0
    rewards = []
    hr_list = []
    map_list = []
    hr_sp_list = []
    map_sp_list = []

    # 强制循环200步，无视 done
    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        # 兼容动作 shape
        if action.shape == (6,):
            action = action.reshape(1, 6)

        obs, reward_vec, done_vec, info = env.step(action)
        reward = reward_vec[0]
        total_reward += reward
        rewards.append(reward)

        # 记录当前状态
        state_hr = env.envs[0].state[0]
        state_map = env.envs[0].state[1]
        hr_list.append(state_hr)
        map_list.append(state_map)

        # 记录当前 setpoint
        current_sp = env.envs[0].setpoints
        hr_sp_list.append(current_sp[0])
        map_sp_list.append(current_sp[1])


    print(f"[{env_label}] Evaluation completed. Total reward: {total_reward}")


    # ---- 返回reward给后面画曲线 ----
    return rewards



if __name__ == "__main__":
    env_types = [
        ('healthy_stable', 'Healthy Stable'),
        ('healthy_exercise', 'Healthy Exercise'),
        ('hypertension_stable', 'Hypertension Stable'),
        ('hypertension_exercise', 'Hypertension Exercise')
    ]

    results = {}
    total_timesteps = 500
    noise_level = 0.1

    for rat_type, label in env_types:
        print(f"\nProcessing environment: {label}")
        log_dir = f"./logs_{rat_type}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        env = initialize_environment(rat_type, noise_level)

        model = train_td3(env, total_timesteps=total_timesteps,
                          log_dir=log_dir, learning_rate=1e-4)

        save_model(model, filename=f"td3_cardiac_model_{rat_type}")

        # 每个环境 评估200步，就能完整看到 "前100步 + 后100步" 的 setpoint 变化
        rewards = evaluate_model(model, env, num_steps=200, env_label=label)
        results[label] = rewards

    # 最后绘制四种环境的 reward 曲线
    plt.figure(figsize=(12, 6))
    colors = ['C0','C1','C2','C3']
    for (label, rews), c in zip(results.items(), colors):
        plt.plot(rews, label=label, color=c)

    plt.xlabel('Evaluation Step')
    plt.ylabel('Reward')
    plt.title('Evaluation Rewards for Different Environments (TD3)')
    plt.legend()
    plt.grid(True)
    plt.show()