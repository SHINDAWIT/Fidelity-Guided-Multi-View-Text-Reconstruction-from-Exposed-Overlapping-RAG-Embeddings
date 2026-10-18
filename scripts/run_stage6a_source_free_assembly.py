#!/usr/bin/env python3

import argparse

from overlap_poc.stage6a_source_free_assembly import (
    load_stage6a_config,
    run_stage6a,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/stage6a_source_free_assembly.yaml"
    )
    args = parser.parse_args()
    result = run_stage6a(load_stage6a_config(args.config))
    print(result["gate"]["state"])


if __name__ == "__main__":
    main()
