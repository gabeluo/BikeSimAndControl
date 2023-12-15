# inputs is [front_steering_angle, a]

import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import cos, sin, tan, radians, degrees, atan2, pi, sqrt
import numpy as np
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
from enum import Enum
import time
from sympy import lambdify, Symbol

t = Symbol("t")


@dataclass
class Bike:
    front_corner_stiff: float
    rear_corner_stiff: float
    mass: float
    inertia: float
    front_length: float
    rear_length: float


@dataclass
class Input:
    delta_dot: float
    a: float


@dataclass
class Point:
    x: float
    y: float
    theta: float


class DynamicStates(Enum):
    X = 0
    X_dot = 1
    Y = 2
    Y_dot = 3
    psi = 4
    psi_dot = 5
    x = 6
    y = 7
    delta = 8


class KinematicStates(Enum):
    x = 0
    y = 1
    psi = 2
    v = 3
    delta = 4


class Simulation:
    def __init__(self, bike: Bike, time_step=0.001, animate=True):
        self.time_step = time_step
        self.bike = bike
        self.xstates, self.ystates = [0], [0]
        self.states = np.array([])
        self.animate = animate
        self.marker_colors = dict(
            markersize=10,
            markerfacecolor="red",
            markerfacecoloralt="lightsteelblue",
            markeredgecolor="brown",
        )

    def update(self, inputs: Input, counter: int, x_func=None, y_func=None):
        # get the new z_dot values
        z_dot = self.find_z_dot(inputs)

        self.states = np.concatenate(
            (self.states, np.array([self.states[:, -1] + z_dot * self.time_step]).T),
            axis=1,
        )

        if self.animate and counter % 10 == 0:
            self.plot(animate=True, x_func=x_func, y_func=y_func)
            plt.pause(0.01)

    def simulate(self, sim_time: int, inputs: Input):
        counter = 0

        while counter < sim_time / self.time_step - 1:
            self.update(inputs=inputs, counter=counter)
            counter += 1

    def get_current_state(self):
        if type(self) is KinematicSim:
            States = KinematicStates
        else:
            States = DynamicStates
        return Point(
            self.states[States.x.value][-1],
            self.states[States.y.value][-1],
            self.states[States.psi.value][-1],
        )

    def plot(self, animate: bool, show=True, x_func=None, y_func=None):
        plt.clf()

        if type(self) is KinematicSim:
            States = KinematicStates
            plt.title("Kinematic Robot Trajectory")
        else:
            States = DynamicStates
            plt.title("Dynamic Robot Trajectory")

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis("equal")
        # plt.axis([-5, 30, -5, 30])
        plt.grid(True)

        if animate:
            plt.plot(
                self.states[States.x.value],
                self.states[States.y.value],
                color="blue",
                label="Robot trajectory",
                markevery=[-1],
                marker=MarkerStyle(
                    "$\Omega$",
                    transform=Affine2D().rotate_deg(
                        degrees(self.states[States.psi.value][-1]) + 90
                    ),
                    capstyle="round",
                ),
                **self.marker_colors
            )

            if x_func:
                lam_x = lambdify(t, x_func, modules=["numpy"])
                lam_y = lambdify(t, y_func, modules=["numpy"])
                t_vals = np.linspace(
                    0, len(self.states[0]) * self.time_step, len(self.states[0])
                )
                x_vals = lam_x(t_vals)
                y_vals = lam_y(t_vals)

                plt.plot(x_vals, y_vals, label="Ideal path", color="orange")
                plt.legend()

        else:
            # plot the path
            plt.plot(
                self.states[States.x.value],
                self.states[States.y.value],
                color="blue",
                label="Robot Trajectory",
            )

            # plot the markers
            MARKER_FREQ = 1 / self.time_step // 2
            # MARKER_FREQ = 1
            for i in range(len(self.states[0])):
                if i % MARKER_FREQ == 0:
                    plt.plot(
                        self.states[States.x.value][i],
                        self.states[States.y.value][i],
                        marker=MarkerStyle(
                            "$\Omega$",
                            transform=Affine2D().rotate_deg(
                                degrees(self.states[States.psi.value][i]) + 90
                            ),
                            capstyle="round",
                        ),
                        **self.marker_colors
                    )

            if x_func:
                lam_x = lambdify(t, x_func, modules=["numpy"])
                lam_y = lambdify(t, y_func, modules=["numpy"])
                t_vals = np.linspace(
                    0, int(len(self.states[0]) * self.time_step), len(self.states[0])
                )
                x_vals = lam_x(t_vals)
                y_vals = lam_y(t_vals)

                plt.plot(x_vals, y_vals, label="Ideal path", color="orange")
                plt.legend()

            if show:
                plt.show()
            if type(self) is KinematicSim:
                print("Final velocity", self.states[States.v.value][-1])
            else:
                print(
                    "Final velocity",
                    sqrt(
                        self.states[States.X_dot.value][-1] ** 2
                        + self.states[States.Y_dot.value][-1] ** 2
                    ),
                )

class DynamicSim(Simulation):
    def __init__(
        self,
        bike: Bike,
        time_step: float,
        animate=True,
        initial_x=0,
        initial_y=0,
        initial_X_dot=0,
        initial_Y_dot=0,
        initial_psi=0,
        initial_delta=0,
    ):
        super().__init__(bike, time_step, animate)
        # (x, x_dot, y, y_dot, psi, psi_dot, X, Y)
        self.states = np.array(
            [
                [0],
                [initial_X_dot],
                [0],
                [initial_Y_dot],
                [initial_psi],
                [0],
                [initial_x],
                [initial_y],
                [initial_delta],
            ]
        )

    # Calculate the z_dot values for the state variables
    def find_z_dot(self, inputs: Input):
        front_theta = (
            (
                self.states[DynamicStates.Y_dot.value][-1]
                + self.bike.front_length * self.states[DynamicStates.psi_dot.value][-1]
            )
            / self.states[DynamicStates.X_dot.value][-1]
            if self.states[DynamicStates.X_dot.value][-1] != 0
            else 0
        )

        rear_theta = (
            (
                self.states[DynamicStates.Y_dot.value][-1]
                - self.bike.rear_length * self.states[DynamicStates.psi_dot.value][-1]
            )
            / self.states[DynamicStates.X_dot.value][-1]
            if self.states[DynamicStates.X_dot.value][-1] != 0
            else 0
        )
        F_f = self.bike.front_corner_stiff * (
            self.states[DynamicStates.delta.value][-1] - front_theta
        )
        F_r = -self.bike.rear_corner_stiff * rear_theta

        # calculate the current z_dot
        return np.array(
            [
                self.states[DynamicStates.X_dot.value][-1],
                (
                    self.states[DynamicStates.psi_dot.value][-1]
                    * self.states[DynamicStates.Y_dot.value][-1]
                    + inputs.a
                ),
                self.states[DynamicStates.Y_dot.value][-1],
                (
                    -self.states[DynamicStates.psi_dot.value][-1]
                    * self.states[DynamicStates.X_dot.value][-1]
                    + 2
                    / self.bike.mass
                    * (F_f * cos(self.states[DynamicStates.delta.value][-1]) + F_r)
                ),
                self.states[DynamicStates.psi_dot.value][-1],
                (
                    2
                    / self.bike.inertia
                    * (self.bike.front_length * F_f - self.bike.rear_length * F_r)
                ),
                self.states[DynamicStates.X_dot.value][-1]
                * cos(self.states[DynamicStates.psi.value][-1])
                - self.states[DynamicStates.Y_dot.value][-1]
                * sin(self.states[DynamicStates.psi.value][-1]),
                self.states[DynamicStates.X_dot.value][-1]
                * sin(self.states[DynamicStates.psi.value][-1])
                + self.states[DynamicStates.Y_dot.value][-1]
                * cos(self.states[DynamicStates.psi.value][-1]),
                inputs.delta_dot,
            ]
        )


class KinematicSim(Simulation):
    def __init__(
        self,
        bike: Bike,
        time_step: float,
        animate=True,
        initial_x=0,
        initial_y=0,
        initial_psi=0,
        initial_v=0,
        initial_delta=0,
    ):
        super().__init__(bike, time_step, animate)
        self.states = np.array(
            [
                [initial_x],
                [initial_y],
                [initial_psi],
                [initial_v],
                [initial_delta],
            ]
        )

        # atan2(2 * (bike.front_length + bike.rear_length), 1)

    # Calculate the z_dot values for the state variables
    def find_z_dot(self, inputs: Input):
        # calculate the current z_dot
        return np.array(
            [
                self.states[KinematicStates.v.value][-1]
                * cos(self.states[KinematicStates.psi.value][-1]),
                self.states[KinematicStates.v.value][-1]
                * sin(self.states[KinematicStates.psi.value][-1]),
                self.states[KinematicStates.v.value][-1]
                / (self.bike.front_length + self.bike.rear_length)
                * tan(self.states[KinematicStates.delta.value][-1]),
                inputs.a,
                inputs.delta_dot,
            ]
        )


def main():
    inputs = Input(delta_dot=radians(1.0), a=1)
    bike = Bike(
        front_corner_stiff=8000,
        rear_corner_stiff=8000,
        mass=2257,
        inertia=3524.9,
        front_length=1.33,
        rear_length=1.616,
    )
    t1 = time.time()
    dynamic_simulation = DynamicSim(bike=bike, time_step=0.01, animate=False)
    dynamic_simulation.simulate(sim_time=25, inputs=inputs)
    dynamic_simulation.plot(animate=False)

    kinematic_simulation = KinematicSim(bike=bike, time_step=0.01, animate=True)
    kinematic_simulation.simulate(sim_time=20, inputs=inputs)
    kinematic_simulation.plot(animate=False)
    print("Total time:", time.time() - t1, "seconds")


if __name__ == "__main__":
    main()
