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


@dataclass
class Point:
    x: float
    y: float
    theta: float


class DynamicStates:
    def __init__(
        self,
        x: float,
        x_dot: float,
        y: float,
        y_dot: float,
        psi: float,
        psi_dot: float,
        X: float,
        Y: float,
    ):
        self.x = x
        self.x_dot = x_dot
        self.y = y
        self.y_dot = y_dot
        self.psi = psi
        self.psi_dot = psi_dot
        self.X = X
        self.Y = Y

    def update_states(self, z_dot):
        return DynamicStates(
            self.x + z_dot[0],
            self.x_dot + z_dot[1],
            self.y + z_dot[2],
            self.y_dot + z_dot[3],
            self.psi + z_dot[4],
            self.psi_dot + z_dot[5],
            self.X,
            self.Y,
        )

    def update_inertial_states(self, Z_dot):
        self.X += Z_dot[0]
        self.Y += Z_dot[1]
        return self


@dataclass
class KinematicStates:
    x: float
    y: float
    psi: float


class Simulation:
    def __init__(self, bike: Bike, time_step=0.001):
        self.time_step = time_step
        self.bike = bike

    def update(self, inputs: Input):
        # get the new z_dot values
        z_dot = self.find_z_dot(inputs)

        # update the state variables
        self.states.append(self.states[-1].update_states(z_dot * self.time_step))

        Z_dot = self.find_Z_dot()

        self.states[-1] = self.states[-1].update_inertial_states(Z_dot * self.time_step)
        # update the inertial state variables

    def get_current_state(self):
        return Point(
            self.states[-1].X,
            self.states[-1].Y,
            self.states[-1].psi,
        )

    def simulate(self, sim_time: int, inputs: Input):
        counter = 0
        while counter < sim_time / self.time_step - 1:
            self.update(inputs)
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
        plt.plot(
            [state.X for state in self.states],
            [state.Y for state in self.states],
            color="blue",
        )

        # plot the markers
        MARKER_FREQ = 1 / self.time_step // 2
        # MARKER_FREQ = 1
        for i in range(len(self.states)):
            if i % MARKER_FREQ == 0:
                plt.plot(
                    self.states[i].X,
                    self.states[i].Y,
                    marker=MarkerStyle(
                        "d",
                        transform=Affine2D().rotate_deg(
                            degrees(self.states[i].psi) - 90
                        ),
                        capstyle="round",
                    ),
                    **marker_colors
                )

        # show plot
        plt.show()


class DynamicSim(Simulation):
    def __init__(self, bike: Bike, time_step: float, initial_x=0, initial_y=0):
        super().__init__(bike, time_step)
        # (x, x_dot, y, y_dot, psi, psi_dot, X, Y)
        self.states = [DynamicStates(0, 0.1, 0, 0, 0, 0, initial_x, initial_y)]

    # Calculate the z_dot values for the state variables
    def find_z_dot(self, inputs: Input):
        front_theta = (
            self.states[-1].y_dot + self.bike.front_length * self.states[-1].psi_dot
        ) / self.states[-1].x_dot
        rear_theta = (
            self.states[-1].y_dot - self.bike.rear_length * self.states[-1].psi_dot
        ) / self.states[-1].x_dot
        F_f = self.bike.front_corner_stiff * (inputs.delta - front_theta)
        F_r = -self.bike.rear_corner_stiff * rear_theta

        # calculate the current z_dot
        return np.array(
            [
                self.states[-1].x_dot,
                (self.states[-1].psi_dot * self.states[-1].y_dot + inputs.a),
                self.states[-1].y_dot,
                (
                    -self.states[-1].psi_dot * self.states[-1].x_dot
                    + 2 / self.bike.mass * (F_f * cos(inputs.delta) + F_r)
                ),
                self.states[-1].psi_dot,
                (
                    2
                    / self.bike.inertia
                    * (self.bike.front_length * F_f - self.bike.rear_length * F_r)
                ),
            ]
        )

    # find the z_dot values for the inertia variables
    def find_Z_dot(self):
        return np.array(
            [
                self.states[-1].x_dot * cos(self.states[-1].psi)
                - self.states[-1].y_dot * sin(self.states[-1].psi),
                self.states[-1].x_dot * sin(self.states[-1].psi)
                + self.states[-1].y_dot * cos(self.states[-1].psi),
            ]
        )


def main():
    inputs = Input(delta=radians(5), a=1)
    bike = Bike(
        front_corner_stiff=6000,
        rear_corner_stiff=6000,
        mass=1235,
        inertia=2200,
        front_length=1,
        rear_length=1,
    )
    simulation = DynamicSim(bike=bike, time_step=0.001)
    simulation.simulate(sim_time=30, inputs=inputs)
    simulation.plot()


if __name__ == "__main__":
    main()
