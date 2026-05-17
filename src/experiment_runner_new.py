"""Compatibility wrapper for the main experiment runner.

The project previously kept two runner files. The maintained implementation is
`src.experiment_runner`; this module re-exports it so old commands continue to
work without carrying a second divergent copy.
"""

try:
    from .experiment_runner import ExperimentConfig, ExperimentRunner, Metrics, main
except ImportError:
    from src.experiment_runner import ExperimentConfig, ExperimentRunner, Metrics, main


__all__ = ["ExperimentConfig", "ExperimentRunner", "Metrics", "main"]


if __name__ == "__main__":
    main()
