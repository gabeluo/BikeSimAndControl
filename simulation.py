# inputs is [front_steering_angle, a]

import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import cos, sin, tan, radians, degrees, atan2, pi, sqrt
import numpy as np
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
import matplotlib.animation as animation


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
        delta: float,
    ):
        self.x = x
        self.x_dot = x_dot
        self.y = y
        self.y_dot = y_dot
        self.psi = psi
        self.psi_dot = psi_dot
        self.X = X
        self.Y = Y
        self.delta = delta

    def update_states(self, z_dot):
        return DynamicStates(
            self.x + z_dot[0],
            self.x_dot + z_dot[1],
            self.y + z_dot[2],
            self.y_dot + z_dot[3],
            self.psi + z_dot[4],
            self.psi_dot + z_dot[5],
            self.X + z_dot[6],
            self.Y + z_dot[7],
            self.delta + z_dot[8],
        )

    def display(self):
        print(
            "x:",
            self.x,
            "x_dot:",
            self.x_dot,
            "y:",
            self.y,
            "y_dot:",
            self.y_dot,
            "psi:",
            self.psi,
            "psi_dot:",
            self.psi_dot,
            "X:",
            self.X,
            "Y:",
            self.Y,
            "delta:",
            self.delta,
        )


class KinematicStates:
    def __init__(
        self,
        x: float,
        y: float,
        psi: float,
        v: float,
        delta: float,
    ):
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.delta = delta

    def update_states(self, z_dot):
        return KinematicStates(
            self.x + z_dot[0],
            self.y + z_dot[1],
            self.psi + z_dot[2],
            self.v + z_dot[3],
            self.delta + z_dot[4],
        )


class Simulation:
    def __init__(self, bike: Bike, time_step=0.001):
        self.time_step = time_step
        self.bike = bike

    def simulate(self, sim_time: int, inputs: Input):
        marker_colors = dict(
            markersize=15,
            markerfacecolor="red",
            markerfacecoloralt="lightsteelblue",
            markeredgecolor="brown",
        )

        fig, ax = plt.subplots()
        ax.set_title("Kinematic Robot Trajectory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid()
        ax.set(xlim=[-5, 50], ylim=[-5, 50])

        # plot the path
        path = ax.plot(
            0,
            0,
            color="blue",
            markevery=[-1],
            # marker=MarkerStyle(
            #     "$\Omega$",
            #     transform=Affine2D().rotate_deg(degrees(self.states[-1].psi) + 90),
            #     capstyle="round",
            # ),
            **marker_colors
        )[0]

        def update(frame):
            # print(x_values[:frame])
            path.set_xdata([state.x for state in self.states])
            path.set_ydata([state.y for state in self.states])
            marker = (3, 2, degrees(self.states[-1].psi) - 90)
            path.set_marker(marker)
            self.update(inputs)
            return path

        anim = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=int(sim_time / self.time_step - 1),
            interval=0.001,
            repeat=False,
        )
        # show plot

        plt.show()


class DynamicSim(Simulation):
    def __init__(self, bike: Bike, time_step: float, initial_x=0, initial_y=0):
        super().__init__(bike, time_step)
        # (x, x_dot, y, y_dot, psi, psi_dot, X, Y)
        self.states = [DynamicStates(0, 0, 0, 0, 0, 0, initial_x, initial_y, 0)]
        self.x_min, self.x_max, self.y_max, self.y_max = 0, 0, 0, 0

    def get_current_state(self):
        return Point(
            self.states[-1].X,
            self.states[-1].Y,
            self.states[-1].psi,
        )

    # Calculate the z_dot values for the state variables
    def find_z_dot(self, inputs: Input):
        front_theta = (
            (self.states[-1].y_dot + self.bike.front_length * self.states[-1].psi_dot)
            / self.states[-1].x_dot
            if self.states[-1].x_dot != 0
            else 0
        )

        rear_theta = (
            (self.states[-1].y_dot - self.bike.rear_length * self.states[-1].psi_dot)
            / self.states[-1].x_dot
            if self.states[-1].x_dot != 0
            else 0
        )
        F_f = self.bike.front_corner_stiff * (self.states[-1].delta - front_theta)
        F_r = -self.bike.rear_corner_stiff * rear_theta

        # calculate the current z_dot
        return np.array(
            [
                self.states[-1].x_dot,
                (self.states[-1].psi_dot * self.states[-1].y_dot + inputs.a),
                self.states[-1].y_dot,
                (
                    -self.states[-1].psi_dot * self.states[-1].x_dot
                    + 2 / self.bike.mass * (F_f * cos(self.states[-1].delta) + F_r)
                ),
                self.states[-1].psi_dot,
                (
                    2
                    / self.bike.inertia
                    * (self.bike.front_length * F_f - self.bike.rear_length * F_r)
                ),
                self.states[-1].x_dot * cos(self.states[-1].psi)
                - self.states[-1].y_dot * sin(self.states[-1].psi),
                self.states[-1].x_dot * sin(self.states[-1].psi)
                + self.states[-1].y_dot * cos(self.states[-1].psi),
                inputs.delta_dot,
            ]
        )

    def update(self, inputs: Input):
        # get the new z_dot values
        z_dot = self.find_z_dot(inputs)

        # update the state variables
        self.states.append(self.states[-1].update_states(z_dot * self.time_step))

    def plot_animate(self, sim_time):
        self.plot()
        marker_colors = dict(
            markersize=10,
            markerfacecolor="red",
            markerfacecoloralt="lightsteelblue",
            markeredgecolor="brown",
        )

        x_values, y_values, psi_values = [], [], []

        for i, state in enumerate(self.states):
            if i % 100 == 0:
                x_values.append(state.X)
                y_values.append(state.Y)
                psi_values.append(state.psi)

        fig, ax = plt.subplots()
        ax.set_title("Kinematic Robot Trajectory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set(xlim=[self.x_min, self.x_max], ylim=[self.y_min, self.y_max])

        # plot the path
        path = ax.plot(
            x_values[0],
            y_values[0],
            color="blue",
            markevery=[-1],
            marker=MarkerStyle(
                "$\Omega$",
                transform=Affine2D().rotate_deg(degrees(psi_values[0]) + 90),
                capstyle="round",
            ),
            **marker_colors
        )[0]

        def update(frame):
            # print(x_values[:frame])
            path.set_xdata(x_values[:frame])
            path.set_ydata(y_values[:frame])
            return path

        anim = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=int(sim_time / self.time_step),
            interval=1,
            repeat=False,
        )
        # show plot
        plt.grid()
        plt.show()

    def plot(self):
        marker_colors = dict(
            markersize=10,
            markerfacecolor="red",
            markerfacecoloralt="lightsteelblue",
            markeredgecolor="brown",
        )
        plt.title("Dynamic Robot Trajectory")
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
                        "$\Omega$",
                        transform=Affine2D().rotate_deg(
                            degrees(self.states[i].psi) + 90
                        ),
                        capstyle="round",
                    ),
                    **marker_colors
                )

        # show plot
        self.x_min, self.x_max = plt.xlim()
        self.y_min, self.y_max = plt.ylim()
        plt.show()


class KinematicSim(Simulation):
    def __init__(self, bike: Bike, time_step: float, initial_x=0, initial_y=0):
        super().__init__(bike, time_step)
        self.states = [
            KinematicStates(
                initial_x,
                initial_y,
                0,
                1,
                0,
            )
        ]
        self.x_min, self.x_max, self.y_max, self.y_max = 0, 0, 0, 0

        # atan2(2 * (bike.front_length + bike.rear_length), 1)

    def get_current_state(self):
        return Point(
            self.states[-1].x,
            self.states[-1].y,
            self.states[-1].psi,
        )

    # Calculate the z_dot values for the state variables
    def find_z_dot(self, inputs: Input):
        # calculate the current z_dot
        return np.array(
            [
                self.states[-1].v * cos(self.states[-1].psi),
                self.states[-1].v * sin(self.states[-1].psi),
                self.states[-1].v
                / (self.bike.front_length + self.bike.rear_length)
                * tan(self.states[-1].delta),
                inputs.a,
                inputs.delta_dot,
            ]
        )

    def update(self, inputs: Input):
        # get the new z_dot values
        z_dot = self.find_z_dot(inputs)

        # update the state variables
        self.states.append(self.states[-1].update_states(z_dot * self.time_step))

    def plot_animate(self):
        self.plot()
        marker_colors = dict(
            markersize=10,
            markerfacecolor="red",
            markerfacecoloralt="lightsteelblue",
            markeredgecolor="brown",
        )

        x_values, y_values, psi_values = [], [], []

        for i, state in enumerate(self.states):
            if i % 10 == 0:
                x_values.append(state.x)
                y_values.append(state.y)
                psi_values.append(state.psi)

        fig, ax = plt.subplots()
        ax.set_title("Kinematic Robot Trajectory")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        print(self.x_max)
        ax.set(xlim=[self.x_min, self.x_max], ylim=[self.y_min, self.y_max])

        # plot the path
        path = ax.plot(
            x_values[0],
            y_values[0],
            color="blue",
            markevery=[-1],
            marker=MarkerStyle(
                "$\Omega$",
                transform=Affine2D().rotate_deg(degrees(psi_values[0]) + 90),
                capstyle="round",
            ),
            **marker_colors
        )[0]

        def update(frame):
            # print(x_values[:frame])
            path.set_xdata(x_values[:frame])
            path.set_ydata(y_values[:frame])
            return path

        anim = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=None,
            interval=1,
            repeat=False,
        )
        # show plot
        ax.grid()
        plt.show()

    def plot(self):
        marker_colors = dict(
            markersize=10,
            markerfacecolor="red",
            markerfacecoloralt="lightsteelblue",
            markeredgecolor="brown",
        )
        plt.title("Kinematic Robot Trajectory")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")

        # plot the path
        plt.plot(
            [state.x for state in self.states],
            [state.y for state in self.states],
            color="blue",
        )

        # plot the markers
        MARKER_FREQ = 1 / self.time_step // 2
        # MARKER_FREQ = 1
        for i in range(len(self.states)):
            if i % MARKER_FREQ == 0:
                plt.plot(
                    self.states[i].x,
                    self.states[i].y,
                    marker=MarkerStyle(
                        "$\Omega$",
                        transform=Affine2D().rotate_deg(
                            degrees(self.states[i].psi) + 90
                        ),
                        capstyle="round",
                    ),
                    **marker_colors
                )

        # show plot
        self.x_min, self.x_max = plt.xlim()
        self.y_min, self.y_max = plt.ylim()
        print("Final velocity", self.states[-1].v)
        plt.grid()
        plt.show()


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
    # dynamic_simulation = DynamicSim(bike=bike, time_step=0.01)
    # dynamic_simulation.simulate(sim_time=0.07, inputs=inputs)
    # dynamic_simulation.plot()

    kinematic_simulation = KinematicSim(bike=bike, time_step=0.01)
    kinematic_simulation.simulate(sim_time=20, inputs=inputs)
    # kinematic_simulation.plot()


if __name__ == "__main__":
    main()
