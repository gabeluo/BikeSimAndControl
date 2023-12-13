import argparse
from controller import Controller
from simulation import (
    DynamicSim,
    KinematicSim,
    DynamicStates,
    KinematicStates,
    Bike,
    Point,
    Input,
)
from sympy import diff, Symbol, sqrt, atan2

t = Symbol("t")


def find_func(bike):
    x = t
    # y = 0.01*(exp(t)-1)
    y = t**2
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
    return theta, v, delta, v_dot, delta_dot


def update_value(v_dot_func, delta_dot_func, time):
    return Input(
        delta_dot=delta_dot_func.subs(t, time).evalf(),
        a=v_dot_func.subs(t, time).evalf(),
    )


def main(args=None):
    # Used only for the point trajectory
    reach_threshold = 0.001
    time_step = 0.001

    bike = Bike(
        front_corner_stiff=8000,
        rear_corner_stiff=8000,
        mass=2257,
        inertia=3524.9,
        front_length=1.33,
        rear_length=1.616,
    )

    # Find the functions for the function trajectory
    if args.trajectory == "function":
        (
            theta_func,
            v_func,
            delta_func,
            v_dot_func,
            delta_dot_func,
        ) = find_func(bike)
    else:
        theta_func = 0
        v_func = 0
        delta_func = 0
        v_dot_func = 0
        delta_dot_func = 0

    if args.sim == "dynamic":
        simulation = DynamicSim(bike=bike, time_step=time_step, animate=True)
        controller = Controller(
            klp=0.5, kap=2.0, kli=0, kai=0, kld=1.0, kad=6, time_step=time_step
        )
    elif args.sim == "kinematic":
        theta = theta_func.subs(t, 0).evalf() if theta_func else 0
        v = v_func.subs(t, 0).evalf() if v_func else 0
        delta = delta_func.subs(t, 0).evalf() if delta_func else 0
        simulation = KinematicSim(
            bike=bike,
            time_step=time_step,
            animate=True,
            initial_x=0,
            initial_y=0,
            initial_psi=theta,
            initial_v=v,
            initial_delta=delta,
        )
        controller = Controller(
            klp=0.1, kap=5, kli=0, kai=0, kld=3, kad=5, time_step=time_step
        )
    else:
        print("This simulation type does not exist.")
        return

    goal_point = Point(args.x, args.y, args.theta)

    time_multiple = 0

    while time_multiple < 0.5 * 10**4:
        reached_goal = (
            True
            if controller.linear_error(simulation.get_current_state(), goal_point)
            < reach_threshold
            else False
        )

        if reached_goal:
            print("reached goal")
            break

        if args.trajectory == "function":
            new_inputs = update_value(
                v_dot_func, delta_dot_func, time_multiple * time_step
            )
        else:
            new_inputs = controller.new_inputs(
                simulation.get_current_state(), goal_point
            )

        simulation.update(new_inputs, time_multiple)

        time_multiple += 1

    simulation.plot(animate=False)
    print(f"The simulation took {time_multiple * simulation.time_step} seconds")


if __name__ == "__main__":
    # Define input arguments
    argParser = argparse.ArgumentParser(description="Point to go towards")
    argParser.add_argument("--x", type=float, default=5)
    argParser.add_argument("--y", type=float, default=5)
    argParser.add_argument("--theta", type=float, default=0)
    argParser.add_argument("--trajectory", type=str, default="function")
    argParser.add_argument("--sim", type=str, default="kinematic")

    args = argParser.parse_args()

    main(args)
