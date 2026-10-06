import os
import sys
import time
import numpy as np
from test_td_info_deepsea20 import test_td_info_deepsea20

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MULTI-SEED CONFIRMATION: HARD SNAPSHOT + TD INFO GAIN (ZERO-MEAN REWARDS)")
    print("=" * 80)
    seeds = [42, 43, 44]
    results = []
    for s in seeds:
        disc, sol, duration = test_td_info_deepsea20(size=20, seed=s, max_episodes=600)
        results.append((s, disc, sol, duration))

    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY TABLE (DEEPSEA N=20, ZERO-MEAN REWARDS)")
    print("=" * 80)
    for s, disc, sol, dur in results:
        print(f"Seed {s} | Discovery: Ep {disc} | Solved: Ep {sol} | Time: {dur:.2f}s")
