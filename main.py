import argparse
from controller import Controller
from simulation import DynamicSim, Bike, Point


def main(args=None):
    reach_threshold = 0.01
    controller = Controller(1.1, 2.0, 0, 0)
    bike = Bike(
        front_corner_stiff=6000,
        rear_corner_stiff=6000,
        mass=1235,
        inertia=2200,
        front_length=1,
        rear_length=1,
    )
    simulation = DynamicSim(bike=bike, time_step=0.001)
    goal_point = Point(args.x, args.y, args.theta)

    time_multiple = 0

    while time_multiple < 10000:
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
        # print("new inputs", new_inputs.delta, new_inputs.a)
        # new_inputs = Input(radians(5), 1)

        simulation.update(new_inputs)

        time_multiple += 1

    simulation.plot()
    print(f"The simulation took {time_multiple * simulation.time_step} seconds")


if __name__ == "__main__":
    # Define input arguments
    argParser = argparse.ArgumentParser(description="Point to go towards")
    argParser.add_argument("--x", type=float, default=1)
    argParser.add_argument("--y", type=float, default=1)
    argParser.add_argument("--theta", type=float, default=0)

    args = argParser.parse_args()

    main(args)
