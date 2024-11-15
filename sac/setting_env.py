from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import torch
import gym
import imageio

def setting(gym_game_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir_path = Path('model')
    model_dir_path.mkdir(parents=True, exist_ok=True)
        
    result_dir_path = Path('result')
    result_dir_path.mkdir(parents=True, exist_ok=True)

    seed = 123456
    torch.manual_seed(seed)
    np.random.seed(seed)

    #env = gym.make(gym_game_name, impact_cost_weight=0.5e-6, render_mode='rgb_array')
    env = gym.make(gym_game_name,  render_mode='rgb_array')
    env.action_space.seed(seed)
    return env, device, seed,model_dir_path,result_dir_path

def random_action_visualization(env, seed, num_steps=100):

    image_dir = Path('images')
    image_dir.mkdir(exist_ok=True)

    images = []

    # 環境をリセット
    obs, info = env.reset(seed=seed)
    for step in tqdm(range(num_steps), desc="Executing random actions"):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # フレームをキャプチャ
        frame = env.render()
        images.append(frame)

        if done:
            obs, info = env.reset()

    gif_path = image_dir / 'random_actions.gif'
    imageio.mimsave(gif_path, images, fps=30)

    env.close()

    print(f"saved gif at : {gif_path}")

if __name__ == "__main__":
    env, device, seed,model_dir_path,result_dir_path = setting()
    random_action_visualization(env, seed, 100)
