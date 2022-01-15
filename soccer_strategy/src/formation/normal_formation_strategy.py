from formation_strategy import FormationStrategy, Formations
import numpy as np

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


