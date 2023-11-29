from simulation import Input, Point
from math import sqrt, atan2, pi


class PID:
    def __init__(self, kp, ki, kd, time_step):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.e_prev = 0
        self.time_step = time_step

    def update(self, error):
        self.integral = self.integral + self.ki * error * self.time_step
        error_dot = (error - self.e_prev) / self.time_step
        self.e_prev = error
        return self.kp * error + self.integral + self.kd * error_dot


# This is only for a point controller implementation
class Controller:
    def __init__(
        self, klp=0.001, kap=1.0, kli=0.0, kai=0.0, kld=0.0, kad=0.0, time_step=0.001
    ):
        self.linear_PID = PID(klp, kli, kld, time_step)
        self.angular_PID = PID(kap, kai, kad, time_step)

    def new_inputs(self, current_pose, goal_pose):
        linear_error = self.linear_error(current_pose, goal_pose)
        angular_error = self.angular_error(current_pose, goal_pose)

        a = self.linear_PID.update(linear_error)
        delta = self.angular_PID.update(angular_error)

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
