"""The stable-ABI extension (pion_embedded._pion); metadata is in pyproject.toml.
One cp311-abi3 wheel serves every CPython >= 3.11."""
from setuptools import Extension, setup

setup(
    ext_modules=[Extension("pion_embedded._pion", ["pion_embedded/_pion.c"], py_limited_api=True)],
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)
