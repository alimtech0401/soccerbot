import math
import numpy as np
import rospy

import config as config
from strategy.strategy import Strategy
from soccer_msgs.msg import GameState
from robot import Robot
from strategy.utils import *
from team import Team
from ball import Ball
from soccer_msgs.msg import GameState

HAVENT_SEEN_THE_BALL_TIMEOUT = 6
GRADIENT_UPDATE_INTERVAL_LENGTH = 0.5

ALPHA = 0.5
BETA = 0.5
EPS = 0.5


class Thresholds:
    POSSESSION = 0.2  # How close player has to be to ball to have possession
    GOALIE_ANGLE = 5  # How close goalie has to be to defense line
    PASS = -1  # Maximum distance between players trying to pass to each other
    OBSTACLE = 1  # Size of player obstacles
    PASSING = 2  # distance for obstacle detection when moving to a position


class DummyStrategy(Strategy):

    def __init__(self):
        self.havent_seen_the_ball_timeout = HAVENT_SEEN_THE_BALL_TIMEOUT
        super(DummyStrategy, self).__init__()


    def update_next_strategy(self, friendly_team: Team, opponent_team: Team, game_state: GameState):
        if friendly_team.average_ball_position.position is not None:
            self.havent_seen_the_ball_timeout = HAVENT_SEEN_THE_BALL_TIMEOUT

            # generate goal pose
            goal_position = config.ENEMY_GOAL_POSITION
            ball = friendly_team.average_ball_position

            if abs(ball.position[1]) < 3.5 and abs(ball.position[0]) < 5:

                current_closest = self.who_has_the_ball(friendly_team.robots, ball)  # Guess who has the ball
                if current_closest is not None:  # and current_closest.send_nav:

                    # generate destination pose
                    ball_position = ball.position
                    player_position = current_closest.position
                    player_angle = current_closest.position[2]

                    diff = ball_position - goal_position
                    diff_unit = diff / np.linalg.norm(diff)
                    diff_angle = math.atan2(-diff_unit[1], -diff_unit[0])

                    distance_of_player_goal_to_ball = 0.1
                    destination_position = ball_position + diff_unit * distance_of_player_goal_to_ball

                    navigation_bias = 1
                    diff = destination_position - player_position
                    # nav bias offset nav goal to be behind the ball
                    destination_position_biased = player_position + diff * navigation_bias

                    # nav goal behind the ball
                    destination_position_biased = [destination_position_biased[0], destination_position_biased[1],
                                                   diff_angle]

                    print("Position of closest player")
                    print(player_position)
                    print("Ball Position")
                    print(ball_position)

                    # difference between robot angle and nav goal angle
                    robot_ball_vector = ball_position - player_position
                    robot_ball_angle = math.atan2(robot_ball_vector[1], robot_ball_vector[0])

                    nav_angle__diff = (player_angle - robot_ball_angle)
                    print("Player angle ", player_angle)
                    print("robot_ball_angle ", robot_ball_angle)
                    print("Angle between player and ball")
                    print(nav_angle__diff)
                    distance_of_player_to_ball = np.linalg.norm(player_position - ball_position)
                    if distance_of_player_to_ball < 0.205 and abs(nav_angle__diff) > 0.15:
                        print("robot ball angle too large, unable to kick " + str(abs(nav_angle__diff)))
                    print("Distance between player and ball")
                    print(distance_of_player_to_ball)
                    if distance_of_player_to_ball < 0.205:
                        if nav_angle__diff > 0.03:
                            # right foot
                            current_closest.kick_with_right_foot = True
                        else:
                            current_closest.kick_with_right_foot = False

                        delta = goal_position - ball.position
                        unit = delta / np.linalg.norm(delta)

                        current_closest.status = Robot.Status.KICKING
                        current_closest.set_kick_velocity(unit * current_closest.max_kick_speed)
                    else:
                        current_closest.set_navigation_position(destination_position_biased)
                        # self.move_player_to(current_closest, destination_position_biased)
        else:
            # If player is not facing the right direction, and not seeing the ball, then face the goal
            self.havent_seen_the_ball_timeout = self.havent_seen_the_ball_timeout - 1
            for player in friendly_team.robots:
                if player.status != Robot.Status.READY:
                    continue

                player_angle = player.position[2]
                player_position = player.position[0:2]

                # Haven't seen the ball timeout
                if self.havent_seen_the_ball_timeout < 0:
                    print("Ball position timeout")
                    rospy.loginfo("Havent seen the ball for a while. Rototating robot " + player.robot_name)
                    self.havent_seen_the_ball_timeout = HAVENT_SEEN_THE_BALL_TIMEOUT
                    turn_position = [player_position[0], player_position[1], player_angle + math.pi]
                    player.set_navigation_position(turn_position)


    def move_player_to(self, player, destination_position):
        # Path planning with obstacle avoidance via potential functions
        # Source:
        # - http://www.cs.columbia.edu/~allen/F17/NOTES/potentialfield.pdf
        obstacles = np.array(player.get_detected_obstacles())
        player_position = np.array(player.position[0:2])
        goal_pos = np.array(destination_position[0:2])

        if distance_between(player_position, goal_pos) > EPS:
            grad = grad_att(ALPHA, player_position, goal_pos)
            if len(obstacles) > 0:
                r_rep = 2 * Thresholds.POSSESSION
                d_rep = float('inf')
                obs_rep = None
                for obs in obstacles:
                    dist = distance_between(obs[0:2], player_position)
                    if dist < d_rep:
                        d_rep = dist
                        obs_rep = obs
                        # grad -= grad_rep(self._beta, r_rep, dist, obs, self._player_pos)
                grad -= grad_rep(BETA, r_rep, d_rep, obs_rep, player_position)
            # Perturb out of local minima
            angle_rand = np.random.uniform(low=-np.pi / 12, high=np.pi / 12)
            rotation_rand = np.array([[np.cos(angle_rand), -np.sin(angle_rand)],
                                      [np.sin(angle_rand), np.cos(angle_rand)]])
            grad_perturbed = rotation_rand @ grad
            # Gradient descent update
            goal_pos = player_position - GRADIENT_UPDATE_INTERVAL_LENGTH * grad_perturbed

        # Update robot state
        diff = player_position - goal_pos
        diff_unit = diff / np.linalg.norm(diff)
        diff_angle = math.atan2(-diff_unit[1], -diff_unit[0])

        # send player to new position
        player.set_navigation_position(np.append(goal_pos, diff_angle))
        print(str(goal_pos) + "  " + str(diff_angle))
