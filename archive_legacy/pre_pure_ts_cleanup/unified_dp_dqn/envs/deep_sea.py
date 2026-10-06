"""Deep Sea Environment matching Osband et al. (2018) / arXiv:1806.03335."""

from typing import Dict, Tuple
import numpy as np


class DeepSeaEnv:
    """Exact N x N Deep Sea Exploration Environment.

    Specifications:
    - Grid size: N rows x N columns.
    - Initial state: (0, 0) (top-left).
    - Transitions: At row r, choosing the right action (action_mapping[r]) moves agent to (r+1, c+1) with cost -0.01/N.
      Choosing wrong action moves agent to (r+1, max(0, c-1)) with cost 0.0.
    - Terminal state: Reaching row N terminates the episode.
    - Treasure: If agent reaches (N-1, N-1) and exits right, it receives +1.0 reward (optimal return = 0.99).
    - Observation: Flattened one-hot vector of dimension N*N.
    """

    def __init__(self, size: int = 10, seed: int = None, randomize_actions: bool = True):
        self.size = int(size)
        self.state_dim = self.size * self.size
        self.action_dim = 2
        self.randomize_actions = randomize_actions
        self.rng = np.random.RandomState(seed)

        # Action mapping: which discrete action (0 or 1) moves right at each row
        if self.randomize_actions:
            self.action_mapping = self.rng.randint(0, 2, size=self.size)
        else:
            self.action_mapping = np.ones(self.size, dtype=np.int64)

        self.row = 0
        self.col = 0
        self.optimal_return = 1.0 - 0.01

    def reset(self) -> np.ndarray:
        self.row = 0
        self.col = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.state_dim, dtype=np.float32)
        idx = self.row * self.size + self.col
        if 0 <= idx < self.state_dim:
            obs[idx] = 1.0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Is this the 'right' action for current row?
        went_right = (action == self.action_mapping[self.row])

        if went_right:
            self.col = min(self.size - 1, self.col + 1)
            reward = -0.01 / float(self.size)
        else:
            self.col = max(0, self.col - 1)
            reward = 0.0

        self.row += 1
        done = (self.row >= self.size)

        # Reached the treasure!
        if done and went_right and self.col == self.size - 1:
            reward += 1.0

        info = {
            "row": self.row,
            "col": self.col,
            "optimal": (went_right and self.col == self.row)
        }

        # Keep row clamped for observation
        if self.row >= self.size:
            self.row = self.size - 1

        return self._get_obs(), float(reward), done, info
