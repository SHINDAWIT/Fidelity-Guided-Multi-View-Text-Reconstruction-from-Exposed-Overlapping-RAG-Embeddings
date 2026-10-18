#!/usr/bin/env python3

import argparse

from overlap_poc.stage6c_evaluate import (
    load_stage6c_evaluation_config,
    run_stage6c_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage6c_evaluate.yaml")
    args = parser.parse_args()
    result = run_stage6c_evaluation(load_stage6c_evaluation_config(args.config))
    print(result["final_gate"]["state"])


if __name__ == "__main__":
    main()
