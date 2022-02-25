#!/usr/bin/env python3
import sys
import time
import os

from isaacgym.torch_utils import torch_rand_float

if "ROS_NAMESPACE" not in os.environ:
    os.environ["ROS_NAMESPACE"] = "/robot1"
import rospy
from play import LaunchModel
import torch

from resources.library.geometry.src.soccer_geometry.transformation import Transformation
from resources.library.pycontrol.src.soccer_pycontrol import soccerbot_controller
from resources.library.trajectories.src.soccer_trajectories import SoccerTrajectoryClass


# useful sudo apt-get install -y python3-rospy
def run_model(obj, env, rsleep):
    n_games = obj.player.games_num
    render = obj.player.render_env
    n_game_life = obj.player.n_game_life
    is_determenistic = obj.player.is_determenistic
    sum_rewards = 0
    sum_steps = 0
    sum_game_res = 0
    n_games = n_games * n_game_life
    games_played = 0
    has_masks = False
    has_masks_func = getattr(env, "has_action_mask", None) is not None

    op_agent = getattr(env, "create_agent", None)
    if op_agent:
        agent_inited = True
        # print('setting agent weights for selfplay')
        # self.player.env.create_agent(self.player.env.config)
        # self.player.env.set_weights(range(8),self.player.get_weights())

    if has_masks_func:
        has_masks = env.has_action_mask()

    need_init_rnn = obj.player.is_rnn
    if games_played < n_games:
        obses = obj.player.env_reset(env)
        batch_size = 1
        batch_size = obj.player.get_batch_size(obses, batch_size)

        if need_init_rnn:
            obj.player.init_rnn()
            need_init_rnn = False

        cr = torch.zeros(batch_size, dtype=torch.float32)
        steps = torch.zeros(batch_size, dtype=torch.float32)

        print_game_res = False

        if steps < obj.player.max_steps:
            if has_masks:
                masks = env.get_action_mask()
                action = obj.player.get_masked_action(
                    obses, masks, is_determenistic)
            else:
                action = obj.player.get_action(obses, is_determenistic)
            obses, r, done, info = obj.player.env_step(env, action)
            cr += r
            steps += 1

            rsleep.sleep()
            # if render:
            #     env.render()
            # env.render(mode='human')
            # time.sleep(0.082)

            all_done_indices = done.nonzero(as_tuple=False)
            done_indices = all_done_indices[::obj.player.num_agents]
            done_count = len(done_indices)
            games_played += done_count

            if done_count > 0:
                if obj.player.is_rnn:
                    for s in obj.player.states:
                        s[:, all_done_indices, :] = s[:,
                                                    all_done_indices, :] * 0.0

                cur_rewards = cr[done_indices].sum().item()
                cur_steps = steps[done_indices].sum().item()

                cr = cr * (1.0 - done.float())
                steps = steps * (1.0 - done.float())
                sum_rewards += cur_rewards
                sum_steps += cur_steps

                game_res = 0.0
                if isinstance(info, dict):
                    if 'battle_won' in info:
                        print_game_res = True
                        game_res = info.get('battle_won', 0.5)
                    if 'scores' in info:
                        print_game_res = True
                        game_res = info.get('scores', 0.5)

                sum_game_res += game_res


def zero_action(env, rsleep):
    action = torch.zeros(env.actions.size(), dtype=torch.float, device=env.device)
    env.step(action)
    rsleep.sleep()

def motor_action(env, rsleep):
    # joint animation states
    ANIM_SEEK_LOWER = 1
    ANIM_SEEK_UPPER = 2
    ANIM_SEEK_DEFAULT = 3
    ANIM_FINISHED = 4

    # initialize animation state
    anim_state = ANIM_SEEK_LOWER
    current_dof = 2
    # check reset
    action = torch.zeros(env.actions.size(), dtype=torch.float, device=env.device)
    action[0][current_dof] = torch_rand_float(-1, 1, (1,1), device=env.device)
    # # animate the dofs
    # speed = 3
    #
    # if anim_state == ANIM_SEEK_LOWER:
    #     action[0][current_dof] -= speed * env.dt
    #     if action[0][current_dof] <= env.dof_pos_limits_lower[current_dof]:
    #         action[0][current_dof] = env.dof_pos_limits_lower[current_dof]
    #         anim_state = ANIM_SEEK_UPPER
    # elif anim_state == ANIM_SEEK_UPPER:
    #     action[0][current_dof] += speed * env.dt
    #     if action[0][current_dof] >= env.dof_pos_limits_upper[current_dof]:
    #         action[0][current_dof] = env.dof_pos_limits_upper[current_dof]
    #         anim_state = ANIM_SEEK_DEFAULT
    # if anim_state == ANIM_SEEK_DEFAULT:
    #     action[0][current_dof] -= speed * env.dt
    #     if action[0][current_dof] <= env.default_dof_pos[0][current_dof]:
    #         action[0][current_dof] = env.default_dof_pos[0][current_dof]
    #         anim_state = ANIM_FINISHED
    # elif anim_state == ANIM_FINISHED:
    #     action[0][current_dof] = env.default_dof_pos[0][current_dof]
    #     # current_dof = (current_dof + 1) % 18
    #     anim_state = ANIM_SEEK_LOWER
    env.step(action)
    rsleep.sleep()

if __name__ == '__main__':

    rospy.init_node("soccer_rlcontrol")
    rospy.logwarn("Loading Config")

    # Testing parameter
    obj = LaunchModel()
    obj.load_config()

    sim_length = 30000
    reset_length = 30000

    rospy.logwarn("Starting Control Loop")

    obj.player = obj.runner.create_player()
    obj.player.restore(obj.runner.load_path)
    env = obj.player.env

    # Trajectory Test
    walker = soccerbot_controller.SoccerbotController(env, 0)
    trajectory_class = SoccerTrajectoryClass(env, 0)
    rsleep = rospy.Rate(1 / 0.00833)  # 0.016667 00833
    # Walk Test
    walker.ready()
    walker.wait(100)
    walker.setGoal(Transformation([2, 0, 0], [0, 0, 0, 1]))

    while not rospy.is_shutdown():
        # run_model(obj, env, rsleep)

        # trajectory_class.run_trajectory("rightkick")
        # walker.run_ros()
        # zero_action(env, rsleep)
        motor_action(env, rsleep)
        pass
