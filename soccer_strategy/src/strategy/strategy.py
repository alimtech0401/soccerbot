import abc
from robot_controlled_3d import RobotControlled3D

from soccer_msgs.msg import GameState
from team import Team

class Strategy():

    @abc.abstractmethod
    def update_next_strategy(self, friendly_team: Team, opponent_team: Team, game_state: GameState):
        raise NotImplementedError

    def get_current_robot(self, friendly_team: Team):
        for robot in friendly_team.robots:
            if robot is RobotControlled3D:
                return robot
        raise AssertionError
