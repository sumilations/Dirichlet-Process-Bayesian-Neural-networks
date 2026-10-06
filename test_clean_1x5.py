import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Load existing metrics & bald maps from previous run or re-plot using saved arrays
# Let's import the data directly from test_1x5_figure
from test_1x5_figure import (
    ground_truth_2d, generate_2d_annulus_data,
    RESULTS_DIR, BRAIN_DIR, SIGMA_ALEATORIC
)

# Re-run rendering with horizontal colorbar under heatmaps
import test_1x5_figure as t5

# Let's inspect test_1x5_figure to just replace the plotting function cleanly
