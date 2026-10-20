#!/usr/bin/env python3
"""Run VectorDBBench 1.0.22's CLI on a StreamingPerformanceCase that carries its parameters.

The 1.0.22 CLI accepts `--case-type StreamingPerformanceCase` but passes no case config for
it (`cli.py: get_custom_case_config` handles only the custom-dataset and filter cases), so
the case runs on its class defaults: Cohere 100K x 768 at 500 rows/s. This wrapper supplies
the config from environment variables and then runs the unchanged CLI in the same process:

  VDBB_STREAM_DATASET   DatasetWithSizeType name or value   (default OpenAISmall)
  VDBB_STREAM_RATE      insert rows/s, a multiple of 100     (default 200)
  VDBB_STREAM_STAGES    JSON list of insert fractions        (default [0.5, 0.8])
  VDBB_STREAM_CONC      JSON list of search concurrencies    (default [5, 10])
  VDBB_STREAM_OPTIMIZE  1|0: optimize() after the last write (default 1)
  VDBB_STREAM_READ_DUR  seconds per concurrency after writes (default 30)

Why 200 rows/s at 50K: the runner gives a stage the time until the next stage, minus the
serial search, minus 15-30 s of slack, split across the concurrencies, and skips the
concurrent search when that is 10 s or less (read_write_runner.py,
get_each_conc_search_dur). At 50K, 2,000 rows/s leaves both in-stream stages without a
concurrent search (they would report recall only); 200 is the highest multiple of 100 at
which both stages run one.

Usage (run with the install's own python, so the right client is imported):
  <venv>/bin/python vdbb_stream.py redis --host localhost --port 6380 --no-ssl --cmd \
      --case-type StreamingPerformanceCase --k 100 --m 16 --ef-construction 128 \
      --ef-runtime 150 --db-label x --task-label x
"""
import json
import os
import sys

from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.cli import cli as vcli
from vectordb_bench.cli.vectordbbench import cli

_orig = vcli.get_custom_case_config


def _dataset(v: str) -> str:
    if v in DatasetWithSizeType.__members__:
        return DatasetWithSizeType[v].value
    return DatasetWithSizeType(v).value


def streaming_config(parameters: dict) -> dict:
    if parameters["case_type"] != "StreamingPerformanceCase":
        return _orig(parameters)
    cfg = {
        "dataset_with_size_type": _dataset(os.environ.get("VDBB_STREAM_DATASET", "OpenAISmall")),
        "insert_rate": int(os.environ.get("VDBB_STREAM_RATE", "200")),
        "search_stages": json.loads(os.environ.get("VDBB_STREAM_STAGES", "[0.5, 0.8]")),
        "concurrencies": json.loads(os.environ.get("VDBB_STREAM_CONC", "[5, 10]")),
        "optimize_after_write": os.environ.get("VDBB_STREAM_OPTIMIZE", "1") == "1",
        "read_dur_after_write": int(os.environ.get("VDBB_STREAM_READ_DUR", "30")),
    }
    print(f"vdbb_stream: StreamingPerformanceCase config {json.dumps(cfg)}", flush=True)
    return cfg


vcli.get_custom_case_config = streaming_config

if __name__ == "__main__":
    sys.argv[0] = "vectordbbench"
    sys.exit(cli())
