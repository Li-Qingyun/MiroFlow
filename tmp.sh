#!/bin/bash

uv run demo_mem0_locomo.py \
    --config_file_name="agent_locomo-gpt5" \
    benchmark.execution.max_concurrent=1 \
    output_dir="./tmp_debug" \
    hydra.run.dir="./tmp_debug"
 