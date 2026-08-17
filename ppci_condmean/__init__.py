"""Paper-facing PPCI conditional-mean utilities."""
from .estimator import (
    fit_ppci_mean,
    ppci_mean_split,
    ppci_mean_nosplit,
    ppci_plus_mean_from_weight_values,
    ppci_plus_mean_given_omegas,
    lo_mean_from_weights,
    ppi_global_mean,
)
from .weights import RKHSLocalizationWeight
from .joint_tuning import JointTuningConfig, tune_joint_from_covariates

__all__ = [
    "RKHSLocalizationWeight",
    "JointTuningConfig",
    "fit_ppci_mean",
    "lo_mean_from_weights",
    "ppi_global_mean",
    "ppci_mean_nosplit",
    "ppci_plus_mean_from_weight_values",
    "ppci_plus_mean_given_omegas",
    "ppci_mean_split",
    "tune_joint_from_covariates",
]
