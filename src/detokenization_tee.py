from argparse import ArgumentParser

from detokenization.step1_save_scores import main as step1
from detokenization.step2_mean_auroc import main as step2

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    step1(num_workers=args.num_workers)
    step2()
