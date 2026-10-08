"""MindSense: early, explainable mental-wellness risk screening.

Academic mini-project package. All heavy logic lives here so notebooks and
the Streamlit app import the same tested code (Rule 7 of the build prompt).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mindsense")
except PackageNotFoundError:  # pragma: no cover - dev checkout
    __version__ = "1.0.0"

GLOBAL_SEED = 42
