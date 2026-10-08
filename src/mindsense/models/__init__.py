"""Trained-model packages (Experiments 3, 4, 6, 7).

Each experiment ships a ``train_<kind>.py`` trainer behind the same
facade, orchestrated by ``mindsense.models.train_all`` (the ``make train``
entry point). Modules are import-light by contract: heavy frameworks
(torch, transformers) are imported lazily inside functions so that a
runtime-only environment never needs them.
"""

__all__: list[str] = []
