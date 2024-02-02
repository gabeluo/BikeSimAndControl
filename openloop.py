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
)
from simulation import Input, Point, Bike


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
        print("Dynamic sim open loop controller functions:")

        # Currently given in the inertia frame of reference, NOT global
        self.x_p = self.t
        print("x_p:", self.x_p)
        self.y_p = sin(self.t)
        # self.y_p = sin(self.t) * exp(-0.5 * self.t)
        print("y_p:", self.y_p)

        self.x = self.x_p
        self.y = self.y_p

        self.x_p_dot = diff(self.x_p)
        print("x_p_dot:", self.x_p_dot)

        self.x_dot = self.x_p_dot

        self.y_p_dot = diff(self.y_p)
        print("y_p_dot", self.y_p_dot)

        self.y_dot = self.y_p_dot / (bike.front_length * bike.mass) - bike.inertia / (
            bike.front_length * bike.mass
        ) * (
            bike.front_length * bike.mass * self.x_p_dot * diff(self.y_p_dot)
            + bike.rear_corner_stiff
            * (bike.front_length + bike.rear_length)
            * self.y_p_dot
        ) / (
            bike.rear_corner_stiff
            * (bike.front_length + bike.rear_length)
            * (bike.inertia - bike.front_length * bike.rear_length * bike.mass)
            * (bike.front_length * bike.mass * self.x_p_dot) ** 2
        )
        print("y_dot:", self.y_dot)

        self.psi_dot = (
            -1
            * (
                bike.front_length**2
                * bike.mass**2
                * self.x_p_dot
                * diff(self.y_p_dot)
                + bike.rear_corner_stiff
                * (bike.front_length + bike.rear_length)
                * (bike.front_length * bike.mass)
                * self.y_p_dot
            )
            / (
                bike.rear_corner_stiff
                * (bike.front_length + bike.rear_length)
                * (bike.inertia - bike.front_length * bike.rear_length * bike.mass)
                + (bike.front_length * bike.mass * self.x_p_dot) ** 2
            )
        )
        print("psi_dot:", self.psi_dot)

        self.psi = integrate(self.psi_dot, self.t)
        print("psi:", self.psi)

        self.x_dot_dot = diff(self.x_dot) - self.psi_dot * self.y_dot
        print("x_dot_dot:", self.x_dot_dot)

        self.delta = (
            diff(self.psi_dot) * bike.inertia / 2
            + bike.rear_length
            * bike.rear_corner_stiff
            * ((bike.rear_length * self.psi_dot - self.y_dot) / self.x_dot)
        ) / (bike.front_length * bike.front_corner_stiff) + (
            self.y_dot + bike.front_length * self.psi_dot
        ) / self.x_dot
        print("delta:", self.delta)

        self.delta_dot = diff(self.delta)
        print("delta_dot:", self.delta_dot)

        # delta, x_dot_dot = symbols("delta x_dot_dot")
        # eq1 = (
        #     -1 * bike.inertia * diff(psi_dot)
        #     + bike.front_length
        #     * bike.front_corner_stiff
        #     * (-1 * ((y_dot) + bike.front_length * psi_dot) / x_dot + delta)
        #     * cos(delta)
        #     + bike.front_length * bike.mass * x_dot_dot * sin(delta)
        #     - bike.rear_length
        #     * bike.rear_corner_stiff
        #     * ((bike.rear_length * psi_dot - y_dot) / x_dot)
        # )
        # eq2 = (
        #     -1 * bike.mass * diff(y_dot)
        #     + bike.mass * x_dot_dot * cos(delta)
        #     - bike.front_corner_stiff
        #     * (-1 * ((y_dot) + bike.front_length * psi_dot) / x_dot + delta)
        #     * sin(delta)
        #     + bike.mass * x_dot_dot
        #     + bike.mass * y_dot * psi_dot
        # )
        # print(solve((eq1, eq2), (delta, x_dot_dot)))

        # delta = symbols('delta')
        # eqn1 = acos(
        #     (
        #         (diff(y_dot) + psi_dot * x_dot) * bike.mass / 2
        #         - bike.rear_corner_stiff
        #         * ((bike.rear_length * psi_dot - y_dot) / x_dot)
        #     )
        #     / (
        #         bike.front_corner_stiff
        #         * (-1 * ((y_dot) + bike.front_length * psi_dot) / x_dot + delta)
        #     )
        # ) - delta

        # print(solve(eqn1, delta))

    def find_initial_vals(self):
        psi = (
            integrate(self.psi_dot, self.t).subs(self.t, 0).evalf()
            if self.psi_dot
            else 0
        )
        psi_dot = self.psi_dot.subs(self.t, 0).evalf() if self.psi_dot else 0
        x_dot = self.x_dot.subs(self.t, 0).evalf() if self.x_dot else 0
        y_dot = self.y_dot.subs(self.t, 0).evalf() if self.y_dot else 0
        delta = self.delta.subs(self.t, 0).evalf() if self.delta else 0
        x_p = self.x_p.subs(self.t, 0).evalf() if self.x_p else 0
        y_p = self.y_p.subs(self.t, 0).evalf() if self.y_p else 0
        print(
            f"Initial values are as follows: psi: {psi}, x_dot: {x_dot}, y_dot: {y_dot}, delta: {delta}"
        )
        return psi, psi_dot, x_dot, y_dot, delta, x_p, y_p

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

    def update_value(self, time):
        return Input(
            delta_dot=self.delta_dot.subs(self.t, time).evalf(),
            a=self.x_dot_dot.subs(self.t, time).evalf(),
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
