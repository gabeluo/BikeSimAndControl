# inputs is [front_steering_angle, a]

import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
from math import cos, sin, radians, degrees
import numpy as np
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle


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
    delta: float
    a: float


class State(Enum):
    x = 0
    x_dot = 1
    y = 2
    y_dot = 3
    psi = 4
    psi_dot = 5


class Simulation:
    def __init__(self, inputs: Input, bike: Bike, time_step=0.001):
        self.time_step = time_step
        inputs.delta = radians(inputs.delta)
        self.inputs = inputs
        self.bike = bike
        # (x, x_dot, y, y_dot, psi, psi_dot)
        self.states = np.zeros([6, 1])
        # (X, Y) (inertial frame, used for plotting)
        self.inertial_states = np.zeros([2, 1])

    def find_z_dot(self, time, inputs: Input):
        pass

    # find the z_dot values for the inertia variables
    def find_Z_dot(self, time):
        return np.array(
            [
                self.states[State.x_dot.value][time]
                * cos(self.states[State.psi.value][time])
                - self.states[State.y_dot.value][time]
                * sin(self.states[State.psi.value][time]),
                self.states[State.x_dot.value][time]
                * sin(self.states[State.psi.value][time])
                + self.states[State.y_dot.value][time]
                * cos(self.states[State.psi.value][time]),
            ]
        )

    def update(self, time: int, inputs: Input):
        # get the new z_dot values
        z_dot = self.find_z_dot(-1, inputs)

        # update the state variables
        self.states = np.concatenate(
            (self.states, np.array([self.states[:, -1] + z_dot * self.time_step]).T),
            axis=1,
        )

        Z_dot = self.find_Z_dot(time - 1)

        # update the inertial state variables
        self.inertial_states = np.concatenate(
            (
                self.inertial_states,
                np.array([self.inertial_states[:, -1] + Z_dot * self.time_step]).T,
            ),
            axis=1,
        )

    def simulate(self, sim_time: int):
        counter = 0
        while counter < sim_time / self.time_step - 1:
            self.update(int(counter), self.inputs)
            counter += 1

    def plot(self):
        marker_colors = dict(
            markersize=10,
            markerfacecolor="red",
            markerfacecoloralt="lightsteelblue",
            markeredgecolor="brown",
        )
        plt.title("Robot Trajectory")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        # plot the path
        plt.plot(self.inertial_states[0], self.inertial_states[1], color="blue")

        # plot the markers
        MARKER_FREQ = int(1 / self.time_step / 2)
        for i in range(len(self.inertial_states[0])):
            if i % MARKER_FREQ == 0:
                plt.plot(
                    self.inertial_states[0][i],
                    self.inertial_states[1][i],
                    marker=MarkerStyle(
                        "d",
                        transform=Affine2D().rotate_deg(
                            degrees(self.states[State.psi.value][i]) - 90
                        ),
                        capstyle="round",
                    ),
                    **marker_colors
                )

        # show plot
        plt.show()


class DynamicSim(Simulation):
    def __init__(self, inputs: Input, bike: Bike, time_step: float):
        super().__init__(inputs, bike, time_step)

    # Calculate the z_dot values for the state variables
    def find_z_dot(self, time: int, inputs: Input):
        front_theta = (
            (
                (
                    self.states[State.y_dot.value][time]
                    + self.bike.front_length * self.states[State.psi_dot.value][time]
                )
                / self.states[State.x_dot.value][time]
            )
            if self.states[State.x_dot.value][time] != 0
            else 0
        )
        rear_theta = (
            (
                (
                    self.states[State.y_dot.value][time]
                    - self.bike.rear_length * self.states[State.psi_dot.value][time]
                )
                / self.states[State.x_dot.value][time]
            )
            if self.states[State.x_dot.value][time] != 0
            else 0
        )
        F_f = self.bike.front_corner_stiff * (inputs.delta - front_theta)
        F_r = -self.bike.rear_corner_stiff * rear_theta

        # calculate the current z_dot
        return np.array(
            [
                self.states[State.x_dot.value][time],
                (
                    self.states[State.psi_dot.value][time]
                    * self.states[State.y_dot.value][time]
                    + inputs.a
                ),
                self.states[State.y_dot.value][time],
                (
                    -self.states[State.psi_dot.value][time]
                    * self.states[State.x_dot.value][time]
                    + 2 / self.bike.mass * (F_f * cos(inputs.delta) + F_r)
                ),
                self.states[State.psi_dot.value][time],
                (
                    2
                    / self.bike.inertia
                    * (self.bike.front_length * F_f - self.bike.rear_length * F_r)
                ),
            ]
        )


def main():
    inputs = Input(delta=5, a=1)
    bike = Bike(
        front_corner_stiff=6000,
        rear_corner_stiff=6000,
        mass=1235,
        inertia=2200,
        front_length=1,
        rear_length=1,
    )
    simulation = DynamicSim(inputs=inputs, bike=bike, time_step=0.001)
    simulation.simulate(sim_time=30)
    simulation.plot()


if __name__ == "__main__":
    main()
