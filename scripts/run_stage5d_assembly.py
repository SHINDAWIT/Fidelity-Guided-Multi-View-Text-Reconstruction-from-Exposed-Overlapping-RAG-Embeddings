#!/usr/bin/env python3

import argparse

from overlap_poc.stage5d_assembly import (
    load_stage5d_assembly_config,
    run_stage5d_assembly,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage5d_assembly.yaml")
    args = parser.parse_args()
    result = run_stage5d_assembly(load_stage5d_assembly_config(args.config))
    print(result["gate"]["state"])


if __name__ == "__main__":
    main()
