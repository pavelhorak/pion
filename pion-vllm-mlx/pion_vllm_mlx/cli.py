"""The `pion-vllm-mlx` command.

    pion-vllm-mlx serve --model mlx-community/Llama-3.2-1B-Instruct-4bit --port 8080
    pion-vllm-mlx --version

`serve` is mlx-lm's OpenAI server with a Pion-backed prompt cache, plus the
Anthropic `/v1/messages` and OpenAI `/v1/responses` endpoints that Claude Code
and Codex speak (see `pion_vllm_mlx.serve`). It needs a running
`pion-server --kvcache` and the `mlx` extra.

`python -m pion_vllm_mlx serve ...` is the same command without the console
script, and `python -m pion_vllm_mlx.serve ...` (serve's flags only) works on
every release that ships serve.py.
"""
from __future__ import annotations

import sys

USAGE = """\
usage: pion-vllm-mlx <command> [options]

commands:
  serve      one local endpoint for coding agents: mlx-lm's server with a
             Pion-backed prompt cache, plus Anthropic /v1/messages (Claude Code)
             and OpenAI /v1/responses (Codex). `pion-vllm-mlx serve --help`
             lists its flags.

options:
  -h, --help     show this message
  -V, --version  print the package version

Needs a running Pion server (`pion-server --kvcache -w 1`) and
`pip install 'pion-vllm-mlx[mlx]'`."""


def version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version as _v
        try:
            return _v("pion-vllm-mlx")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    return "unknown (not installed as a package)"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if args[0] in ("-V", "--version"):
        print(f"pion-vllm-mlx {version()}")
        return 0
    cmd, rest = args[0], args[1:]
    if cmd == "serve":
        try:
            import mlx_lm  # noqa: F401
        except ImportError as e:
            print(f"pion-vllm-mlx serve needs mlx-lm ({e}).\n"
                  "Install it with:  pip install 'pion-vllm-mlx[mlx]'   (Apple Silicon only)",
                  file=sys.stderr)
            return 1
        from pion_vllm_mlx.serve import main as serve_main
        return serve_main(rest) or 0
    print(f"pion-vllm-mlx: unknown command {cmd!r}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
