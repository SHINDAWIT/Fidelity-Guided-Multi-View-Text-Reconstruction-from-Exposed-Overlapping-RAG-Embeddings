#!/usr/bin/env python3
from __future__ import annotations

import argparse

from overlap_poc.stage3_longdoc_feasibility import (
    load_stage3a_config,
    run_stage3a_feasibility,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tokenizer-only Stage 3A long-document feasibility scan"
    )
    parser.add_argument(
        "--config", default="configs/stage3a_longdoc_feasibility.yaml"
    )
    args = parser.parse_args()
    result = run_stage3a_feasibility(load_stage3a_config(args.config))
    print(f"status={result['status']}")
    print(f"gate={result['gate']['state']}")
    print(f"selected_max_k={result['gate']['selected_max_k']}")
    for k in result["k_values"]:
        profile = result["profiles"][str(k)]
        print(
            f"K={k} L={profile['required_content_tokens']} "
            f"eligible={profile['length_eligible_passages']} "
            f"stable={profile['all_windows_roundtrip_stable_passages']} "
            f"acceptance={profile['acceptance_yield_among_length_eligible']:.4f} "
            f"dedup={profile['deduplication']['deduplicated_stable_documents']}"
        )


if __name__ == "__main__":
    main()
