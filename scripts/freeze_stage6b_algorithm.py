#!/usr/bin/env python3

import argparse

from overlap_poc.stage6b_freeze import load_stage6b_config, run_stage6b_freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage6b_algorithm_freeze.yaml")
    args = parser.parse_args()
    result = run_stage6b_freeze(load_stage6b_config(args.config))
    print(result["status"])


if __name__ == "__main__":
    main()
