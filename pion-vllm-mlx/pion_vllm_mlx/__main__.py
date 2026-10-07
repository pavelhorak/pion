"""`python -m pion_vllm_mlx serve ...` — the `pion-vllm-mlx` command without the console script."""
import sys

from pion_vllm_mlx.cli import main

sys.exit(main())
