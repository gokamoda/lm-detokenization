from argparse import ArgumentParser

from empirical import (
    six_terms_importance_main,
    six_terms_importance_vis,
    vs_tptpp_main,
    vs_tptpp_vis,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["vs_tptpp", "six"])
    parser.add_argument("--func", type=str, choices=["vis", "main"])
    args = parser.parse_args()

    if args.mode == "vs_tptpp":
        if args.func == "main":
            vs_tptpp_main()
        elif args.func == "vis":
            vs_tptpp_vis()
        else:
            raise ValueError("Invalid func")
    elif args.mode == "six":
        if args.func == "main":
            six_terms_importance_main()
        elif args.func == "vis":
            six_terms_importance_vis()
        else:
            raise ValueError("Invalid func")
