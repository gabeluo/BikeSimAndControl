import argparse
from controller import Controller
from simulation import DynamicSim, KinematicSim, Bike, Point, Input
from sympy import *

t = Symbol("t")
a = Symbol("a")
b = Symbol("b")


def find_func(bike):
    x = t
    # y = 0.01*(exp(t)-1)
    y = t**3
    theta = atan2(diff(y), diff(x))
    print("theta", theta)
    v = sqrt(diff(x) ** 2 + diff(y) ** 2)
    print("v", v)
    delta = atan2((bike.front_length + bike.rear_length) * diff(theta), v)
    print("delta", delta)
    v_dot = diff(v)
    delta_dot = diff(delta)
    print("delta_dot", delta_dot)
    return v_dot, delta_dot


def update_value(v_dot_func, delta_dot_func, time):
    return Input(
        delta_dot=delta_dot_func.subs(t, time).evalf(),
        a=v_dot_func.subs(t, time).evalf(),
    )


def main(args=None):
    reach_threshold = 0.001
    time_step = 0.001
    # Dynamic Controller (use default x and y position)
    controller = Controller(
        klp=1.1, kap=2.0, kli=0, kai=0, kld=0, kad=0, time_step=time_step
    )

    # Kinematic Controller (use 5, 10 for x and y)
    # controller = Controller(
    #     klp=0.1, kap=5, kli=0, kai=0, kld=3, kad=5, time_step=time_step
    # )
    bike = Bike(
        front_corner_stiff=8000,
        rear_corner_stiff=8000,
        mass=2257,
        inertia=3524.9,
        front_length=1.33,
        rear_length=1.616,
    )
    if args.sim == "dynamic":
        simulation = DynamicSim(bike=bike, time_step=time_step)
    elif args.sim == "kinematic":
        simulation = KinematicSim(bike=bike, time_step=time_step)
    else:
        print("This simulation type does not exist.")
        return

    goal_point = Point(args.x, args.y, args.theta)

    time_multiple = 0

    v_dot_func, delta_dot_func = find_func(bike)

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

        new_inputs = controller.new_inputs(simulation.get_current_state(), goal_point)
        # new_inputs = update_value(v_dot_func, delta_dot_func, time_multiple * time_step)
        # print(new_inputs.delta_dot)

        simulation.update(new_inputs)
        # print(new_inputs)
        # print(simulation.states[-1].x, simulation.states[-1].y)

        time_multiple += 1

    simulation.plot()
    print(f"The simulation took {time_multiple * simulation.time_step} seconds")


if __name__ == "__main__":
    # Define input arguments
    argParser = argparse.ArgumentParser(description="Point to go towards")
    argParser.add_argument("--x", type=float, default=5)
    argParser.add_argument("--y", type=float, default=5)
    argParser.add_argument("--theta", type=float, default=0)
    argParser.add_argument("--sim", type=str, default="dynamic")

    args = argParser.parse_args()

    main(args)
