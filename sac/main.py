import torch
from model import SoftActorCriticModel
from memory import ReplayMemory
from setting_env import setting
import numpy as np
from pathlib import Path
import imageio


args = {
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.2,
    'seed': 123456,
    'batch_size': 10,
    'hidden_size': 256,
    'start_steps': 50,
    'updates_per_step': 1,
    'max_steps' : 100,
    'target_update_interval': 1,
    'memory_size': 100,
    'epochs': 10000,
    'eval_interval': 10
}

if __name__ == "__main__":

    #gym_game_name = "HumanoidStandup-v5"
    gym_game_name = "Pendulum-v1"

    env, device, seed,model_dir_path,result_dir_path = setting(gym_game_name)
    agent = SoftActorCriticModel(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
        action_scale=env.action_space.high[0], args=args, device=device
    )
    memory = ReplayMemory(args['memory_size'])

    episode_reward_list = []
    eval_reward_list = []

    n_steps = 0
    n_update = 0
    for i_episode in range(1, args['epochs'] + 1):

        episode_reward = 0
        done = False
        state = env.reset()
        state = np.array(state)

        for _ in range(args['max_steps']):
            
            if args['start_steps'] > n_steps:
                action = env.action_space.sample()
            else:
                if state.shape[0] == env.observation_space.shape[0]:
                    action = agent.select_action(state)
                else:
                    action = env.action_space.sample()


            if len(memory) > args['batch_size']:
                agent.update_parameters(memory, args['batch_size'], n_update)
                n_update += 1

            next_state, reward, done, _,info = env.step(action)
            next_state = np.array(next_state)
            n_steps += 1
            episode_reward += reward

            if state.shape == next_state.shape:# env reset cause unmatch of the shape
                memory.push(state=state, action=action, reward=reward, next_state=next_state, mask=float(not done))

            state = next_state
            if done:
                break

        episode_reward_list.append(episode_reward)
        if i_episode % args['eval_interval'] == 0:
            avg_reward = 0.
            images = []  # Initialize the list to store frames

            # Collect images during the first evaluation episode
            state = env.reset()
            state = np.array(state)
            episode_reward = 0
            done = False
            print("eval start")
            for _ in range(args['max_steps']):
                print(f"state shape;{state.shape[0]}")
                print(f"observation shape;{env.observation_space.shape[0]}")
                with torch.no_grad():
                    if state.shape[0] == env.observation_space.shape[0]:
                        action = agent.select_action(state, evaluate=True)
                    else:
                        action = env.action_space.sample()
                next_state, reward, done, _, info = env.step(action)
                next_state = np.array(next_state)
                episode_reward += reward
                state = next_state

                # Capture frame
                frame = env.render()
                images.append(frame)
                if done:
                    break

            avg_reward += episode_reward


            avg_reward /= args['eval_interval']
            eval_reward_list.append(avg_reward)

            print("Episode: {}, Eval Avg. Reward: {:.0f}".format(i_episode, avg_reward))

             # Save images as a GIF
            image_dir = Path('images')
            image_dir.mkdir(exist_ok=True)
            gif_path = image_dir / f'{gym_game_name}:episode_{i_episode}.gif'
            imageio.mimsave(gif_path, images, fps=30)
            print(f"Saved GIF at: {gif_path}")

    print('Game Done !! Max Reward: {:.2f}'.format(np.max(eval_reward_list)))

    torch.save(agent.actor_net.to('cpu').state_dict(), model_dir_path.joinpath(f'{gym_game_name}_sac_actor.pth'))