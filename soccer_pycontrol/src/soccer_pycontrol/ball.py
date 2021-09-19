import os
import numpy as np
from os.path import expanduser

if os.getenv('ENABLE_PYBULLET', True):
    import pybullet as p


class Ball:

    def __init__(self, position=(0, 0, 0), velocity=[0, 0]):
        self.position = position
        self.velocity = velocity
        home = expanduser("~")
        if os.environ['USER'] == 'shahryar':
            home = home + "/hdd"

        self.path = home + "/catkin_ws/src/soccerbot/soccer_description/models/Ball/ball.urdf"
        if os.getenv('ENABLE_PYBULLET', True):
            self.ball = p.loadURDF(self.path, basePosition=self.position,
                                   baseOrientation=[0, 0, 0, 1])
            p.changeDynamics(self.ball, linkIndex=-1, restitution=0.99)

    def set_position(self, position):
        if os.getenv('ENABLE_PYBULLET', True):
            p.removeBody(self.ball)
        self.__init__(position, self.velocity)

    def set_velocity(self, velocity):
        if os.getenv('ENABLE_PYBULLET', True):
            p.removeBody(self.ball)
        self.__init__(self.position, velocity)

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.ball)
        return np.array(pos, dtype=np.float32)[0:2]

    def get_velocity(self):
        return self.velocity
