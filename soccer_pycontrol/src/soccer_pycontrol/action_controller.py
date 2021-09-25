import os
from transformation import Transformation
from soccerbot_rl import SoccerbotRl
from ramp import Ramp
import pybullet as pb
import pybullet_data
from ball import Ball
from time import sleep
import time
import numpy as np
import gym
from ray.rllib.utils.framework import try_import_tf
import matplotlib.pyplot as plt

tf1, tf, tfv = try_import_tf()
import ray
import ray.rllib.agents.ars as ars

env_id = "norm-v0"
checkpoint_path = "/home/manx52/catkin_ws/src/soccerbot/soccer_pycontrol/src/soccer_rlcontrol/results/Kick/checkpoint-7280"


class ActionController:
    PYBULLET_STEP = 0.004

    def __init__(self):
        pb.connect(pb.GUI)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        pb.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=0,
                                      cameraTargetPosition=[0, 0, 0.25])
        pb.setGravity(0, 0, -9.81)
        self.ramp = Ramp("plane_implicit.urdf", (0, 0, 0), (0, 0, 0), lateralFriction=0.9, spinningFriction=0.9,
                         rollingFriction=0.0)
        self.soccerbot = SoccerbotRl(Transformation())
        self.ball = Ball()
        self.velocity_configuration = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.foot_pressure_values = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.RNN = [0]
        print("RAY INIT START")
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
        self.agent = trainer(env="gym_soccerbot:norm-v0", config=config)
        self.agent.load_checkpoint(checkpoint_path)
        self.env = gym.make(env_id, renders=True, env_name="gym_soccerbot:kick-v0", goal=[0, 0.3])
        self.try_once = True
        print("RAY INIT END")

    def run(self, action: str):
        if action == "kick":
            self.run_kick()
        pass

    def get_kick_state_vector(self):
        # get from self.soccerbot and self.ball
        np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
        positions = self.soccerbot.joints_pos()[:16]  # Array of floats
        print(f'positions: {positions}')
        velocities = self.soccerbot.joints_vel()[:16]  # Array of floats
        print(f'velocities: {velocities}')
        imu = self.soccerbot.get_imu_raw()  # IMU values
        print(f'imu: {imu}')

        feet_pressure_sensors = self.soccerbot.get_foot_pressure_sensors_RL(self.ramp.plane)
        print(f'feet_pressure_sensors: {feet_pressure_sensors}')
        orientation_vector = self.soccerbot.off_orn()
        print(f'orientation_vector: {orientation_vector}')
        ball_pos = self.ball.get_position()
        print(f'ball_pos: {ball_pos}')
        rnn = self.RNN
        print(f'rnn: {rnn}')

        # Concatenate vector
        # return np.concatenate(
        #     (positions, velocities, imu_angvel, imu_linacc, imu_orientation, orientation_vector, feet_pressure_sensors))
        return np.concatenate(
            (positions, velocities, imu, orientation_vector, feet_pressure_sensors, ball_pos, rnn))

    def wait(self, steps):
        for i in range(steps):
            time.sleep(ActionController.PYBULLET_STEP)
            if os.getenv('ENABLE_PYBULLET', True):
                pb.stepSimulation()

    def run_kick(self):
        # Written by shahryar
        # state_vector = self.get_kick_state_vector()
        t = 0
        completed = False
        velocity = []
        obs = []
        obs3 = []
        while t <= 0.9:
            self.soccerbot.get_imu_raw()
            if os.getenv('ENABLE_PYBULLET', True):
                pb.stepSimulation()
            t = t + ActionController.PYBULLET_STEP
            sleep(ActionController.PYBULLET_STEP)

        # t = 0
        while t <= 5:
            if not completed:
                observation_vector = self.get_kick_state_vector()
                joint_angles = observation_vector[:16]
                observation_vector = self.env.normalize(observation_vector, self.env.env.observation_limit_low,
                                                        self.env.env.observation_limit_high,
                                                        self.env.observation_plus_range)
                # clip
                observation_vector = np.clip(observation_vector, -10.0, 10.0)

                action_vector = self.agent.compute_action(observation_vector)
                # action_vector = self.env.denormalize(action_vector, self.env.env.action_space.low,
                #                                      self.env.env.action_space.high,
                #                                      self.env.action_plus_range)
                self.RNN = action_vector[16:]
                pb.saveBullet('hello')
                self.soccerbot.motor_control(action_vector[:16], joint_angles, self.env.env)
            if os.getenv('ENABLE_PYBULLET', True):
                pb.stepSimulation()
            t = t + ActionController.PYBULLET_STEP
            sleep(ActionController.PYBULLET_STEP)
            obs.append(self.soccerbot.get_imu_raw())
            obs3.append(observation_vector)
            velocity.append(self.soccerbot.joints_pos()[:16])
        obs_real = np.array(obs3)
        obs2 = np.array(obs)
        velocity2 = np.array(velocity)
        x_axis = np.linspace(0, t, num=len(velocity2))

        fig, ax = plt.subplots(2, 2)
        # Left Leg
        ax[0, 0].plot(x_axis, velocity2[:, 4], label='LEFT_LEG_1')
        ax[0, 0].plot(x_axis, velocity2[:, 5], label='LEFT_LEG_2')
        ax[0, 0].plot(x_axis, velocity2[:, 6], label='LEFT_LEG_3')
        ax[0, 0].plot(x_axis, velocity2[:, 7], label='LEFT_LEG_4')
        ax[0, 0].plot(x_axis, velocity2[:, 8], label='LEFT_LEG_5')
        ax[0, 0].plot(x_axis, velocity2[:, 9], label='LEFT_LEG_6')
        ax[0, 0].set_title('Angle Left Foot')
        ax[0, 0].set_xlabel('time (t)')
        ax[0, 0].set_ylabel('Angle')
        ax[0, 0].legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

        # Right leg
        ax[0, 1].plot(x_axis, velocity2[:, 10], label='RIGHT_LEG_1')
        ax[0, 1].plot(x_axis, velocity2[:, 11], label='RIGHT_LEG_2')
        ax[0, 1].plot(x_axis, velocity2[:, 12], label='RIGHT_LEG_3')
        ax[0, 1].plot(x_axis, velocity2[:, 13], label='RIGHT_LEG_4')
        ax[0, 1].plot(x_axis, velocity2[:, 14], label='RIGHT_LEG_5')
        ax[0, 1].plot(x_axis, velocity2[:, 15], label='RIGHT_LEG_6')
        ax[0, 1].set_title('Angle Right Foot')
        ax[0, 1].set_xlabel('time (t)')
        ax[0, 1].set_ylabel('Angle')
        ax[0, 1].legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

        # Arms
        ax[1, 0].plot(x_axis, velocity2[:, 0], label='LEFT_ARM_1')
        ax[1, 0].plot(x_axis, velocity2[:, 1], label='LEFT_ARM_2')
        ax[1, 0].plot(x_axis, velocity2[:, 2], label='RIGHT_ARM_1')
        ax[1, 0].plot(x_axis, velocity2[:, 3], label='RIGHT_ARM_2')
        ax[1, 0].set_title('Angle Arms')
        ax[1, 0].set_xlabel('time (t)')
        ax[1, 0].set_ylabel('Angle')
        ax[1, 0].legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

        # OBS imu
        ax[1, 1].plot(x_axis, obs2[:, 0], label='IMU X')
        ax[1, 1].plot(x_axis, obs2[:, 1], label='IMU Y')
        ax[1, 1].plot(x_axis, obs2[:, 2], label='IMU Z')
        ax[1, 1].set_title('IMU')
        ax[1, 1].set_xlabel('time (t)')
        ax[1, 1].set_ylabel('Angles')
        ax[1, 1].legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

        plt.show()
