from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()
from time import sleep
import gym
import ray
from ray import tune
import ray.rllib.agents.ars as ars
from matplotlib import pyplot as plt

import cProfile, pstats, io
from pstats import SortKey
import numpy as np
checkpoint_path = "/home/robosoccer/shah_ws/rl-locomotion-engine/results/PBT-Kick/2021-07-06_05:00:20.451493_pbt_gym_soccerbot:norm-v0_seed1234_/ARS_gym_soccerbot:norm-v0_9a00d_00002_2_noise_stdev=0.047732,sgd_stepsize=0.042573_2021-07-06_05-00-25/checkpoint_006440/checkpoint-6440"
checkpoint_path = "/home/robosoccer/shah_ws/rl-locomotion-engine/results/PBT-Kick/2021-07-10_00:10:47.186992_pbt_gym_soccerbot:norm-v0_seed1234_/ARS_gym_soccerbot:norm-v0_d06b1_00002_2_noise_stdev=0.017331,sgd_stepsize=0.0084534_2021-07-10_00-10-51/checkpoint_030800/checkpoint-30800"
checkpoint_path = "/home/manx52/catkin_ws/src/soccerbot/soccer_pycontrol/src/soccer_rlcontrol/results/Kick/checkpoint-7280"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

if tf.test.gpu_device_name():
    print('Default GPU Device Details: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install Tensorflow that supports GPU")
if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    ray.init(local_mode=True)
    trainer, trainer_class = ars.ARSTrainer, ars
    # load
    config = trainer_class.DEFAULT_CONFIG.copy()
    config["framework"] = "tf"
    config["eager_tracing"] = False
    config["env_config"] = {"env_name": "gym_soccerbot:kick-v0"}
    config["num_workers"] = 1
    config["model"] = {"fcnet_hiddens": [128, 128]}
    config["num_gpus"] = 0
    agent = trainer(env="gym_soccerbot:norm-v0", config=config)
    agent.load_checkpoint(checkpoint_path)
    #agent.restore(checkpoint_path)
    #agent.get_policy().get_weights()
    #agent.get_policy().export_model("/home/shahryar/PycharmProjects/DeepRL/weights")


    # instantiate env class
    env_id = "norm-v0"
    env = gym.make(env_id, renders=True, env_name="gym_soccerbot:kick-v0", goal=[0, 0.3], horizon=180)

    #pr = cProfile.Profile()
    #pr.enable()
    # ... do something ...
    # run until episode ends
    # for j in range(100):

    while True:
        episode_reward = 0
        done = False
        obs = env.reset()
        #print(obs)
        i = 0
        obs_list = []
        t_list = []
        while not done:
            print(obs)
            obs_list.append(obs)
            t_list.append(i * 0.0082)

            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            # print(f'Z: {env.env._global_pos()[2]:.3f}')
            sleep(0.0082)
            episode_reward += reward
            i += 1
            print(f'step: {i} | reward: {reward} | state: {info["end_cond"]}')
        print(f'episode_reward: {episode_reward:.3f}, episode_len: {i}, info: {info}')

        # obs_np = np.array(obs_list)
        # t_np = np.array(t_list)
        # plt.plot(t_np, obs_np[:, 32:35])
        # plt.show()

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    env.close()
    ray.shutdown()
