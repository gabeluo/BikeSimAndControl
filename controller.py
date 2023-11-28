from simulation import State, Input, Point
from dataclasses import dataclass
from math import sqrt, atan2, pi


# This is only for a point controller implementation
class Controller:
    def __init__(self, klp=0.001, kap=1.0, ki=0.0, kd=0.0):
        self.klp = klp
        self.kap = kap
        self.ki = ki
        self.kd = kd

    def new_inputs(self, current_pose, goal_pose):
        linear_error = self.linear_error(current_pose, goal_pose)
        angular_error = self.angular_error(current_pose, goal_pose)

        # print(linear_error, angular_error)

        a = self.klp * linear_error
        delta = self.kap * angular_error

        return Input(delta, a)

    def linear_error(self, current_pose, goal_pose: Point):
        return sqrt(
            (current_pose.x - goal_pose.x) ** 2 + (current_pose.y - goal_pose.y) ** 2
        )

    def angular_error(self, current_pose, goal_pose: Point):
        error_angular = (
            atan2(goal_pose.y - current_pose.y, goal_pose.x - current_pose.x)
            - current_pose.theta
        )

        if error_angular <= -pi:
            error_angular += 2 * pi

        elif error_angular >= pi:
            error_angular -= 2 * pi

        return error_angular


def main():
    controller = Controller()
    print(controller)


if __name__ == "__main__":
    main()
