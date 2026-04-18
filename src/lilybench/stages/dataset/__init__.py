"""Dataset preparation and loading utilities.

Submodules are imported explicitly by consumers — keeping this ``__init__``
free of eager imports means modules that don't need ``torch`` (e.g. the
dataset builder) can be used without the training extras installed.
"""
