from sympy import (
    diff,
    Symbol,
    sqrt,
    sin,
    atan2,
    cos,
    solve,
    symbols,
    acos,
    integrate,
    sqrt,
    exp,
    pi,
    Eq,
    lambdify,
    nsolve,
)
from scipy.optimize import fsolve
from simulation import Input, Point, Bike
import numpy as np


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
    def __init__(self, bike, time_step):
        pass

    def find_initial_vals(self):
        psi = self.initial_theta.subs(self.t, 0).evalf() if self.initial_theta else 0
        psi_dot = (
            self.initial_theta_dot.subs(self.t, 0).evalf()
            if self.initial_theta_dot
            else 0
        )
        x_dot = self.x_dot.subs(self.t, 0).evalf() if self.x_dot else 0
        y_dot = self.y_dot.subs(self.t, 0).evalf() if self.y_dot else 0
        delta = 0
        print(
            f"Initial values are as follows: psi: {psi}, x_dot: {x_dot}, y_dot: {y_dot}, delta: {delta}"
        )
        return psi, psi_dot, x_dot, y_dot, delta

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

    def update_value(self, time, bike):

        delta_0 = (
            (self.local_y_dot + (bike.front_length + bike.rear_length) * self.theta_dot)
            - self.x_dot
            * self.theta_ddot
            * -1
            * bike.inertia
            / 2
            / (bike.front_length + bike.rear_length)
            / bike.front_corner_stiff
        ) / (
            (self.local_y_dot + (bike.front_length + bike.rear_length) * self.theta_dot)
            * self.theta_ddot
            * -1
            * bike.inertia
            / 2
            / (bike.front_length + bike.rear_length)
            / bike.front_corner_stiff
            + self.x_dot
        )

        delta = nsolve(
            (self.eqn.subs(self.t, time)),
            (self.delta),
            (delta_0.subs(self.t, time).evalf()),
        )

        self.states.append(delta)

        return Input(
            delta_dot=(self.states[-1] - self.states[-2]) / self.time_step,
            a=self.a.subs(self.t, time).evalf(),
        )


def main():
    bike = Bike(
        front_corner_stiff=8000,
        rear_corner_stiff=8000,
        mass=2257,
        inertia=3524.9,
        front_length=1.33,
        rear_length=1.616,
    )
    dynamic = DynamicOLController(bike, 0.01)


if __name__ == "__main__":
    main()
