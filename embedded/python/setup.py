"""The stable-ABI extension (pionlite._pion); metadata is in pyproject.toml.
One cp311-abi3 wheel serves every CPython >= 3.11."""
from setuptools import Extension, setup

setup(
    ext_modules=[Extension("pionlite._pion", ["pionlite/_pion.c"], py_limited_api=True)],
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)
