import argparse

def main(args=None):

    def simulate(self):
    counter = 0
    while counter < self.sim_time / self.time_step - 1:
        self.update(int(counter))
        counter += 1

    args.x, args.y, args.theta))


if __name__ == "__main__":
    # Define input arguments
    argParser = argparse.ArgumentParser(description="Point to go towards")
    argParser.add_argument("--x", type=float, default=None)
    argParser.add_argument("--y", type=float, default=None)
    argParser.add_argument("--theta", type=float, default=None)

    args = argParser.parse_args()

    main(args)