"""EigenTune: parameter-efficient fine-tuning inside a model's pretrained singular subspaces."""

__version__ = "0.2.0.dev0"

from .compat import EigenTunedLayer
from .config import EigenTuneConfig
from .layers import EigenTuneLinear
from .model import adapter_report, get_eigentune_model, merge_adapter, print_trainable_parameters, unmerge_adapter
from .serialization import AdapterMismatchError, load_adapter, save_adapter

__all__ = [
    "AdapterMismatchError",
    "EigenTuneConfig",
    "EigenTunedLayer",
    "EigenTuneLinear",
    "adapter_report",
    "get_eigentune_model",
    "load_adapter",
    "merge_adapter",
    "print_trainable_parameters",
    "save_adapter",
    "unmerge_adapter",
]
