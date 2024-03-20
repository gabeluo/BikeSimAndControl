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
        super().__init__(time_step)

        self.delta = Symbol("delta")
        self.a = Symbol("a")
        self.theta = Symbol("theta")
        self.theta_dot = Symbol("theta_dot")

        print("Dynamic sim open loop controller functions:")

        # Currently given in the inertia frame of reference, NOT global
        self.x = self.t
        print("x:", self.x)

        self.x_dot = diff(self.x)
        print("x_dot", self.x_dot)

        self.x_ddot = diff(self.x_dot)
        print("x_ddot", self.x_ddot)

        self.y = self.t
        print("y:", self.y)

        self.y_dot = diff(self.y)
        print("y_dot", self.y_dot)

        self.y_ddot = diff(self.y_dot)
        print("y_ddot:", self.y_ddot)

        self.initial_theta = atan2(self.y_dot, self.x_dot)
        self.initial_theta_dot = diff(self.initial_theta)

        # self.theta = atan2(self.y_dot, self.x_dot)
        # print("theta:", self.theta)

        # self.theta_dot = diff(self.theta)
        # print("theta_dot:", self.theta_dot)

        # self.theta_ddot = diff(self.theta_dot)
        # print("theta_ddot", self.theta_ddot)

        self.local_x_dot = cos(self.theta) * self.x_dot + sin(self.theta) * self.y_dot
        self.local_y_dot = -sin(self.theta) * self.x_dot + cos(self.theta) * self.y_dot

        alpha_f = atan2(
            (self.local_y_dot + (bike.front_length + bike.rear_length) * self.theta_dot)
            * cos(self.delta)
            - self.x_dot * sin(self.delta),
            (
                (
                    self.local_y_dot
                    + (bike.front_length + bike.rear_length) * self.theta_dot
                )
                * sin(self.delta)
                + self.x_dot * cos(self.delta)
            ),
        )
        alpha_r = atan2(self.local_y_dot, self.local_x_dot)

        self.local_x_ddot = self.theta_dot * self.local_y_dot
        self.local_y_ddot = (
            -self.theta_dot * self.local_x_dot
            + 2 / bike.mass * (-1 * bike.front_corner_stiff * alpha_f) * cos(self.delta)
            + bike.rear_corner_stiff * alpha_r
        )

        self.eqn1 = (
            self.x_ddot
            - cos(self.theta) * self.local_x_ddot
            + -1 * sin(self.theta) * self.theta_dot * self.local_x_dot
        ) / cos(self.theta) - self.a

        self.eqn2 = (
            self.y_ddot
            - sin(self.theta) * self.local_y_ddot
            + cos(self.theta) * self.theta_dot * self.local_y_dot
        ) / sin(self.theta) - self.a

        # self.a = (
        #     self.x_ddot
        #     - cos(self.theta) * self.local_x_ddot
        #     + -1 * sin(self.theta) * self.theta_dot * self.local_x_dot
        # ) / cos(self.theta)

        # self.eqn = (
        #     alpha_f * cos(self.delta)
        #     - self.theta_ddot
        #     * -1
        #     * bike.inertia
        #     / 2
        #     / (bike.front_length + bike.rear_length)
        #     / bike.front_corner_stiff
        # )

        # self.states = [0]

        self.theta_ddot = (
            2
            / bike.inertia
            * (bike.front_length + bike.rear_length)
            * -1
            * bike.front_corner_stiff
            * alpha_f
            * cos(self.delta)
        )

        # theta, theta_dot, delta, a
        self.states = np.array(
            [
                [self.initial_theta.subs(self.t, 0)],
                [self.initial_theta_dot.subs(self.t, 0)],
                [0],
                [0],
            ]
        )

        # print(nsolve(
        #     (self.eqn1.subs([(self.t, 1), (self.theta, 0.495), (self.theta_dot, -0.651)]), self.eqn2.subs([(self.t, 1), (self.theta, 0.495), (self.theta_dot, -0.651)])),
        #     (self.a, self.delta),
        #     (self.states[3][-1], self.states[2][-1]),
        # ))

        # equations = [eqn1.subs(self.t, 1), eqn2.subs(self.t, 1)]
        # equations_func = [lambdify((self.a, self.delta), eq) for eq in equations]

        # initial_guess = [0.1, 0.1]

        # solution = fsolve([eq(*vars) for eq in equations_func], initial_guess)

        # solution = solve((eqn1.subs(self.t, 1), eqn2.subs(self.t, 1)), (self.delta, self.a))
        # print("Solution", solution)

        # self.x_p_dot = diff(self.x_p)
        # print("x_p_dot:", self.x_p_dot)

        # self.x_dot = self.x_p_dot

        # self.y_p_dot = diff(self.y_p)
        # print("y_p_dot", self.y_p_dot)

        # self.y_dot = self.y_p_dot / (bike.front_length * bike.mass) - bike.inertia / (
        #     bike.front_length * bike.mass
        # ) * (
        #     bike.front_length * bike.mass * self.x_p_dot * diff(self.y_p_dot)
        #     + bike.rear_corner_stiff
        #     * (bike.front_length + bike.rear_length)
        #     * self.y_p_dot
        # ) / (
        #     bike.rear_corner_stiff
        #     * (bike.front_length + bike.rear_length)
        #     * (bike.inertia - bike.front_length * bike.rear_length * bike.mass)
        #     * (bike.front_length * bike.mass * self.x_p_dot) ** 2
        # )
        # print("y_dot:", self.y_dot)

        # self.psi_dot = (
        #     -1
        #     * (
        #         bike.front_length**2 * bike.mass**2 * self.x_p_dot * diff(self.y_p_dot)
        #         + bike.rear_corner_stiff
        #         * (bike.front_length + bike.rear_length)
        #         * (bike.front_length * bike.mass)
        #         * self.y_p_dot
        #     )
        #     / (
        #         bike.rear_corner_stiff
        #         * (bike.front_length + bike.rear_length)
        #         * (bike.inertia - bike.front_length * bike.rear_length * bike.mass)
        #         + (bike.front_length * bike.mass * self.x_p_dot) ** 2
        #     )
        # )
        # print("psi_dot:", self.psi_dot)

        # self.psi = integrate(self.psi_dot, self.t)
        # print("psi:", self.psi)

        # self.x_dot_dot = diff(self.x_dot) - self.psi_dot * self.y_dot
        # print("x_dot_dot:", self.x_dot_dot)

        # self.delta = (
        #     diff(self.psi_dot) * bike.inertia / 2
        #     + bike.rear_length
        #     * bike.rear_corner_stiff
        #     * ((bike.rear_length * self.psi_dot - self.y_dot) / self.x_dot)
        # ) / (bike.front_length * bike.front_corner_stiff) + (
        #     self.y_dot + bike.front_length * self.psi_dot
        # ) / self.x_dot
        # print("delta:", self.delta)

        # self.delta_dot = diff(self.delta)
        # print("delta_dot:", self.delta_dot)

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

        # delta_0 = (
        #     (self.local_y_dot + (bike.front_length + bike.rear_length) * self.theta_dot)
        #     - self.x_dot
        #     * self.theta_ddot
        #     * -1
        #     * bike.inertia
        #     / 2
        #     / (bike.front_length + bike.rear_length)
        #     / bike.front_corner_stiff
        # ) / (
        #     (self.local_y_dot + (bike.front_length + bike.rear_length) * self.theta_dot)
        #     * self.theta_ddot
        #     * -1
        #     * bike.inertia
        #     / 2
        #     / (bike.front_length + bike.rear_length)
        #     / bike.front_corner_stiff
        #     + self.x_dot
        # )

        # delta = nsolve(
        #     (self.eqn.subs(self.t, time)),
        #     (self.delta),
        #     (delta_0.subs(self.t, time).evalf()),
        # )

        # self.states.append(delta)

        # return Input(
        #     delta_dot=(self.states[-1] - self.states[-2]) / self.time_step,
        #     a=self.a.subs(self.t, time).evalf(),
        # )

        result = nsolve(
            (
                self.eqn1.subs(
                    [
                        (self.t, time),
                        (self.theta, self.states[0][-1]),
                        (self.theta_dot, self.states[1][-1]),
                    ]
                ),
                self.eqn2.subs(
                    [
                        (self.t, time),
                        (self.theta, self.states[0][-1]),
                        (self.theta_dot, self.states[1][-1]),
                    ]
                ),
            ),
            (self.a, self.delta),
            (self.states[3][-1], self.states[2][-1]),
        )

        a = result[0]
        delta = result[1]

        theta_ddot = self.theta_ddot.subs(
            [
                (self.t, time),
                (self.a, a),
                (self.delta, delta),
                (self.theta, self.states[0][-1]),
                (self.theta_dot, self.states[1][-1]),
            ]
        )

        self.states = np.concatenate(
            (
                self.states,
                np.array(
                    [
                        self.states[:, -1]
                        + np.array([self.states[1][-1], theta_ddot, 0, 0])
                        * self.time_step
                    ]
                ).T,
            ),
            axis=1,
        )
        self.states[2][-1] = delta
        self.states[3][-1] = a

        print(
            Input(
                delta_dot=(self.states[2][-1] - self.states[2][-2]) / self.time_step,
                a=a,
            )
        )

        return Input(
            delta_dot=(self.states[2][-1] - self.states[2][-2]) / self.time_step,
            a=a,
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
