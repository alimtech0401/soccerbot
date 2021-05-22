import math
import numpy as np
from robot import Robot
from robot_ros import RobotRos

class Strategy:
    def __init__(self):
        pass

    def update_both_team_strategy(self, robots, ball):
        friendly = []
        opponent = []
        for robot in robots:
            if robot.team == Robot.Team.FRIENDLY or robot.Team == RobotRos.Team.FRIENDLY:
                friendly.append(robot)
            else:
                opponent.append(robot)

        self.update_next_strategy(friendly, ball)
        self.update_next_strategy(opponent, ball)

    def update_next_strategy(self, robots, ball):
        raise NotImplementedError


class DummyStrategy(Strategy):
    def __init__(self, RobotClass):
        self.RobotClass = RobotClass

    def who_has_the_ball(self, robots, ball):
        closest_dist = math.inf
        current_closest = None
        for robot in robots:
            dist = np.linalg.norm(ball.get_position()[0:2] - robot.get_position()[0:2])
            if dist < closest_dist:
                closest_dist = dist
                current_closest = robot
        return current_closest

    def update_next_strategy(self, robots, ball):
        # Guess who has the ball
        current_closest = self.who_has_the_ball(robots, ball)

        a = current_closest.get_position()
        b = ball.get_position()
        if np.linalg.norm(current_closest.get_position()[0:2] - ball.get_position()) < 0.2:
            # Stop moving
            current_closest.set_navigation_position(current_closest.get_position())

            if current_closest.team == self.RobotClass.Team.FRIENDLY:
                opponent_goal = np.array([0, 4.5])
            else:
                opponent_goal = np.array([0, -4.5])

            # Kick the ball towards the goal
            delta = opponent_goal - ball.get_position()
            unit = delta / np.linalg.norm(delta)

            current_closest.set_kick_velocity(unit * current_closest.max_kick_speed)
            current_closest.status = self.RobotClass.Status.KICKING
        else:
            current_closest.set_navigation_position(np.append(ball.get_position(), 0))
            current_closest.status = self.RobotClass.Status.WALKING
