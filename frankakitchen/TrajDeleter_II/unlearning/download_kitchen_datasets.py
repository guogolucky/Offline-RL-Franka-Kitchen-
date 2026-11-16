import argparse
import d4rl
import gym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kitchen-complete-v0")
    args = parser.parse_args()

    datasets = [
        "kitchen-complete-v0",
        "kitchen-partial-v0",
        "kitchen-mixed-v0",
    ]

    for dataset in datasets:
        env = gym.make(dataset)
        env.get_dataset()


if __name__ == "__main__":
    main()