from sympy import diff, Symbol, sqrt, sin, atan2
from simulation import Input, Point


class OLController:
    def __init__(self, time_step):
        # Define symbolic variable to be used in later calculations
        self.t = Symbol("t")
        self.time_step = time_step

    def update_value(self, time):
        return Input(
            delta_dot=self.delta_dot.subs(self.t, time).evalf(),
            a=self.v_dot.subs(self.t, time).evalf(),
        )

    def get_position_funcs(self):
        return self.x, self.y


class KinematicOLController(OLController):
    def __init__(self, bike, time_step):
        super().__init__(time_step)
        print("Kinematic sim open loop controller functions:")
        self.x = self.t
        print("x:", self.x)
        # y = 0.01*(exp(t)-1)
        self.y = sin(self.t)
        print("y:", self.y)
        self.theta = atan2(diff(self.y), diff(self.x))
        print("theta:", self.theta)
        self.v = sqrt(diff(self.x) ** 2 + diff(self.y) ** 2)
        print("v:", self.v)
        self.delta = atan2(
            (bike.front_length + bike.rear_length) * diff(self.theta), self.v
        )
        print("delta:", self.delta)
        self.v_dot = diff(self.v)
        print("v_dot:", self.v_dot)
        self.delta_dot = diff(self.delta)
        print("delta_dot:", self.delta_dot)

    def find_initial_vals(self):
        theta = self.theta.subs(self.t, 0).evalf() if self.theta else 0
        v = self.v.subs(self.t, 0).evalf() if self.v else 0
        delta = self.delta.subs(self.t, 0).evalf() if self.delta else 0
        return theta, v, delta

    def find_next_point(self, look_ahead, time_multiple):
        return Point(
            x=self.x.subs(
                self.t, (time_multiple + look_ahead) * self.time_step
            ).evalf(),
            y=self.y.subs(
                self.t, (time_multiple + look_ahead) * self.time_step
            ).evalf(),
            # Theta value for point is unused since the current controller cannot
            # orient the robot to the desired heading at the goal point
            theta=0,
        )


class DynamicOLController(OLController):
    def find_func(self, bike):
        x = self.t
        print("x:", x)
        # y = 0.01*(exp(t)-1)
        y = sin(self.t)
        print("y:", y)
        theta = atan2(diff(y), diff(x))
        print("theta:", theta)
        v = sqrt(diff(x) ** 2 + diff(y) ** 2)
        print("v:", v)
        delta = atan2((bike.front_length + bike.rear_length) * diff(theta), v)
        print("delta:", delta)
        v_dot = diff(v)
        print("v_dot:", v_dot)
        delta_dot = diff(delta)
        print("delta_dot:", delta_dot)
        return x, y, theta, v, delta, v_dot, delta_dot
