#!/usr/bin/env python3

import argparse

from overlap_poc.stage6c_strong import (
    load_stage6c_strong_config,
    run_stage6c_strong_generation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage6c_strong.yaml")
    args = parser.parse_args()
    result = run_stage6c_strong_generation(load_stage6c_strong_config(args.config))
    print(result["status"])


if __name__ == "__main__":
    main()
