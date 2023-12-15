import argparse
from controller import Controller
from simulation import (
    DynamicSim,
    KinematicSim,
    Bike,
    Point,
    Input,
)
from sympy import diff, Symbol, sqrt, sin, atan2

t = Symbol("t")


def find_func(bike):
    x = t
    print("x:", x)
    # y = 0.01*(exp(t)-1)
    y = sin(t)
    print("y:", y)
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
    return x, y, theta, v, delta, v_dot, delta_dot


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
            x_func,
            y_func,
            theta_func,
            v_func,
            delta_func,
            v_dot_func,
            delta_dot_func,
        ) = find_func(bike)
    else:
        x_func = None
        y_func = None
        theta_func = None
        v_func = None
        delta_func = None
        v_dot_func = None
        delta_dot_func = None

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
            klp=0.01, kap=1.1, kli=0, kai=0, kld=11, kad=8, time_step=time_step
            # klp=0.1, kap=5, kli=0, kai=0, kld=3, kad=5, time_step=time_step

        )
    else:
        print("This simulation type does not exist.")
        return

    goal_point = Point(args.x, args.y, args.theta)

    time_multiple = 0

    while time_multiple < 7 * (1/time_step):
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
            new_inputs_open_loop = update_value(
                v_dot_func, delta_dot_func, time_multiple * time_step
            )
            new_inputs_closed_loop = controller.new_inputs(
                simulation.get_current_state(),
                Point(
                    x=x_func.subs(t, (time_multiple+1) * time_step).evalf(),
                    y=y_func.subs(t, (time_multiple+1) * time_step).evalf(),
                    theta=theta_func.subs(t, (time_multiple+1) * time_step).evalf(),
                ),
            )
            open_loop_factor = 0.975
            closed_loop_factor = 1 - open_loop_factor
            new_inputs = Input(
                (
                    open_loop_factor * new_inputs_open_loop.delta_dot
                    + closed_loop_factor * new_inputs_closed_loop.delta_dot
                ),
                (
                    open_loop_factor * new_inputs_open_loop.a
                    + closed_loop_factor * new_inputs_closed_loop.a
                ),
            )
        else:
            new_inputs = controller.new_inputs(
                simulation.get_current_state(), goal_point
            )

        simulation.update(new_inputs, time_multiple, x_func, y_func)

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
    argParser.add_argument("--trajectory", type=str, default="trajectory")
    argParser.add_argument("--sim", type=str, default="kinematic")

    args = argParser.parse_args()

    main(args)
