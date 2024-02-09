import argparse
from controller import Controller
from simulation import (
    DynamicSim,
    KinematicSim,
    Bike,
    Point,
    Input,
)
from openloop import KinematicOLController, DynamicOLController
from math import pi


def main(args=None):
    # Used only for the point trajectory
    reach_threshold = 0.001

    # Time step for the sim and controller
    time_step = 0.001

    # How long to run the function for
    duration = 2

    # look_ahead distance when using the closed-loop controller
    look_ahead = 25

    # Bike parameters
    # Note that only the lengths are used for the kinematic simulation
    bike = Bike(
        front_corner_stiff=8000,
        rear_corner_stiff=8000,
        mass=2257,
        inertia=3524.9,
        front_length=1.33,
        rear_length=1.616,
    )

    # bike = Bike(
    #     front_corner_stiff=14000,
    #     rear_corner_stiff=15000,
    #     mass=990,
    #     inertia=1800,
    #     front_length=1.25,
    #     rear_length=1.3,
    # )

    if args.sim == "dynamic":
        olcontroller = DynamicOLController(bike=bike, time_step=time_step)
        psi, psi_dot, x_dot, y_dot, delta = olcontroller.find_initial_vals()
        simulation = DynamicSim(
            bike=bike,
            time_step=time_step,
            animate=True,
            # global x and y position
            # initial_x=0,
            # initial_y=0,
            # # local x_dot and y_dot
            # initial_X_dot=1.0,
            # initial_Y_dot=0.0,
            # # heading with respect to global frame
            # initial_psi=pi / 4,
            # initial_delta=0,
            # global x and y position
            initial_x=0,
            initial_y=0,
            # # local x_dot and y_dot
            initial_X_dot=x_dot,
            initial_Y_dot=y_dot,
            # # heading with respect to global frame
            initial_psi=psi,
            initial_delta=delta,
        )
        controller = Controller(
            klp=0.001, kap=0.1, kli=0, kai=0, kld=0.001, kad=0.02, time_step=time_step
        )
    elif args.sim == "kinematic":
        olcontroller = KinematicOLController(bike=bike, time_step=time_step)
        theta, v, delta = olcontroller.find_initial_vals()
        simulation = KinematicSim(
            bike=bike,
            time_step=time_step,
            animate=True,
            initial_x=-0.2,
            initial_y=-0.2,
            initial_psi=theta,
            initial_v=v,
            initial_delta=delta,
        )
        controller = Controller(
            klp=0.01, kap=10, kli=0, kai=0, kld=0.05, kad=45, time_step=time_step
        )
        # Note that the gain for acceleration should be 10 times smaller than the gain for delta_dot
    else:
        print("This simulation type does not exist.")
        return

    # Find the functions for the function trajectory
    if args.trajectory == "function":
        x_func, y_func = olcontroller.get_position_funcs()
    else:
        x_func, y_func = None, None

    goal_point = Point(args.x, args.y, args.theta)

    time_multiple = 0

    while time_multiple < duration * (1 / time_step):
        if args.trajectory == "point":
            reached_goal = (
                True
                if controller.linear_error(simulation.get_current_state(), goal_point)
                < reach_threshold
                else False
            )

            if reached_goal:
                print("Reached goal!")
                break

        if args.trajectory == "function":
            new_inputs_open_loop = olcontroller.update_value(time_multiple * time_step, bike)
            # new_inputs_closed_loop = controller.new_inputs(
            #     simulation.get_current_state(),
            #     olcontroller.find_next_point(look_ahead, time_multiple),
            # )

            # Give the closed-loop controller equal weight to the open-loop
            open_loop_factor = 1
            closed_loop_factor = 0
            new_inputs = Input(
                (
                    open_loop_factor * new_inputs_open_loop.delta_dot
                    + closed_loop_factor * 0.0
                ),
                (
                    open_loop_factor * new_inputs_open_loop.a
                    + closed_loop_factor * 0.0
                ),
            )
        else:
            new_inputs = controller.new_inputs(
                simulation.get_current_state(), goal_point
            )

        simulation.update(
            inputs=new_inputs, counter=time_multiple, x_func=x_func, y_func=y_func
        )

        time_multiple += 1

    simulation.plot(animate=False, show=True, x_func=x_func, y_func=y_func)

    # lam_x = lambdify(t, x_func, modules=["numpy"])
    # lam_y = lambdify(t, y_func, modules=["numpy"])
    # t_vals = linspace(0, int(time_multiple * simulation.time_step), time_multiple)
    # x_vals = lam_x(t_vals)
    # y_vals = lam_y(t_vals)

    # plt.plot(x_vals, y_vals, label="Ideal path", color="orange")
    # plt.legend()
    # plt.show()

    print(f"The simulation took {time_multiple * simulation.time_step} seconds")


if __name__ == "__main__":
    # Define input arguments
    argParser = argparse.ArgumentParser(description="Point to go towards")
    argParser.add_argument("--x", type=float, default=5)
    argParser.add_argument("--y", type=float, default=5)
    argParser.add_argument("--theta", type=float, default=0)
    argParser.add_argument("--trajectory", type=str, default="function")
    argParser.add_argument("--sim", type=str, default="dynamic")

    args = argParser.parse_args()

    main(args)
