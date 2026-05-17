"""Backward-compatible comparison script.

The original script compared ETA-based and IL+RL. It now includes the third
parallel method: Multi-Agent RL. You can also run compare_all_methods.py, which
has the same evaluation logic with a clearer filename.
"""

from compare_all_methods import main


if __name__ == "__main__":
    main()
