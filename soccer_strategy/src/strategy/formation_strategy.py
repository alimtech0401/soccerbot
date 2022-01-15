import math
import numpy as np
import rospy
import enum

import config as config
from strategy.strategy import Strategy
# from soccer_msgs.msg import GameState
from robot import Robot


class FieldPosition:
    # can add other stuff like shape later
    def __init__(self, center):
        self.center = center


class FieldPositions:
    CENTER_STRIKER = FieldPosition(np.array([0, 0]))
    GOALIE = FieldPosition(np.array([4.5, 0]))
    RIGHT_WING = FieldPosition(np.array([-2, 3]))
    LEFT_WING = FieldPosition(np.array([-2, -3]))
    RIGHT_BACK = FieldPosition(np.array([3.5, 2]))
    LEFT_BACK = FieldPosition(np.array([3.5, -2]))
    CENTER_BACK = FieldPosition(np.array([3, 0]))


# formations can be in a human readable data file?
class Formation:
    def __init__(self, positions):
        # array where 0 position is position for robot 0 (goalie) I guess?
        #TODO: change positions to dictionary with robot_id
        self.positions = positions

    def closest_position(self, target):
        a = [distance(position.center, target) for position in self.positions]
        return np.argmin(a) + 1


# should have a utils class for geometry calculations, geometry file/class?
def distance(o1, o2):
    return np.linalg.norm(o1 - o2)


class Formations:
    #don't go to role
    DEFENSIVE = Formation(
        [FieldPositions.GOALIE, FieldPositions.LEFT_BACK, FieldPositions.RIGHT_BACK, FieldPositions.CENTER_BACK])
    ATTACKING = Formation(
        [FieldPositions.GOALIE, FieldPositions.LEFT_WING, FieldPositions.CENTER_STRIKER, FieldPositions.RIGHT_WING])


# class FormationSet:
#     def __init__(self, formations):
#         self.formations = formations
#
#     def decide_formation(self, robots, ball, game_properties):
#         raise NotImplementedError("Please Implement this method")
#
# class NormalFormationSet(FormationSet):
#     def __init__(self):
#         super()

class FormationStrategy:

    def __init__(self, default_formation, team_data):
        self.formation = default_formation
        self.team_data = team_data

    def update_next_strategy(self, friendly, opponent, ball, game_properties):
        for robot in friendly:
            if robot.role == Robot.Role.GOALIE:
                self.formation = self.decide_formation(friendly, ball, game_properties)
            self.act_individual(robot)

    def update_strategy(self, robot):
        self.formation = self.decide_formation()
        self.act_individual(robot)

    def decide_formation(self):
        raise NotImplementedError("please implement")

    def act_individual(self, robot):
        raise NotImplementedError("please implement")


class NormalFormationStrategy(FormationStrategy):
    def __init__(self, team_data):
        super().__init__(Formations.DEFENSIVE, team_data)

    # something so there can be different formation deciders like a super defensive biased one
    def decide_formation(self):
        if self.team_data.ball.is_known() and self.team_data.ball.position[0] < 0:  # ball in oppents side
            return Formations.ATTACKING
        else:
            return Formations.DEFENSIVE

    # this can be something to help implement more decisions in a better way?
    def act_individual(self, robot):
        # a decision maker class so these can be strung together and more advacned oeprations?
        if self.team_data.ball.is_known() and robot.robot_id == self.formation.closest_position(self.team_data.ball.position):
            # have actions class with subclasses
            # robot.do_now(Actions.MOVE_TO, [ball])
            #print("robot ", robot.robot_id, " will move to ball")
            goal = [self.team_data.ball.position[0], self.team_data.ball.position[1], 3.14]
            if not np.allclose(goal, robot.goal_position):
                robot.set_navigation_position(goal)
            return
        else:
            # robot.do_now(Actions.MOVE_TO, [self.formation.positions[robot.robot_id])
            #print("robot ", robot.robot_id, " will move to position")
            goal = [self.formation.positions[robot.robot_id-1].center[0], self.formation.positions[robot.robot_id -1].center[1], 3.14]
            #TODO add this if to robot.set_navigation_position
            if not np.allclose(goal, robot.goal_position):
                robot.set_navigation_position(goal)
            return


