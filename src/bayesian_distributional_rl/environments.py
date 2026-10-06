"""
Tabular MDP Environments for Bayesian Distributional RL Benchmarking
1. RiverSwim (Strehl & Littman, 2008): Exploration benchmark with sparse delayed reward
2. CliffWalking (Sutton & Barto): Grid navigation with catastrophic penalty region
"""

import numpy as np
from typing import Tuple, Optional


class RiverSwimEnv:
    """
    RiverSwim Environment (Strehl & Littman, 2008; Osband et al., 2013).
    States: 0, 1, ..., N-1.
    Start: State 0.
    Actions:
        0: Swim Left (with current) - deterministic transition to max(s - 1, 0).
           Small reward r = 0.05 at s=0.
        1: Swim Right (against current):
           - In state 0: P(s=0)=0.4, P(s=1)=0.6. Reward = 0.
           - In state s (1 <= s <= N-2): P(s-1)=0.1, P(s)=0.6, P(s+1)=0.3. Reward = 0.
           - In state N-1: P(N-2)=0.4, P(N-1)=0.6. Reward = 1.0!
    """
    def __init__(self, n_states: int = 6, max_steps: int = 40, seed: Optional[int] = None):
        self.n_states = n_states
        self.num_states = n_states
        self.num_actions = 2
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        
        self.current_state = 0
        self.current_step = 0

    def reset(self) -> int:
        self.current_state = 0
        self.current_step = 0
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        self.current_step += 1
        s = self.current_state
        r = 0.0
        
        if action == 0:
            # Swim left (with current)
            if s == 0:
                s_next = 0
                r = 0.05
            else:
                s_next = s - 1
                r = 0.0
        else:
            # Swim right (against current)
            if s == 0:
                s_next = 1 if self.rng.random() < 0.6 else 0
                r = 0.0
            elif s == self.n_states - 1:
                # Reached the treasure chest!
                if self.rng.random() < 0.6:
                    s_next = self.n_states - 1
                    r = 1.0
                else:
                    s_next = self.n_states - 2
                    r = 0.0
            else:
                p = self.rng.random()
                if p < 0.1:
                    s_next = s - 1
                elif p < 0.7:
                    s_next = s
                else:
                    s_next = s + 1
                r = 0.0
                
        self.current_state = s_next
        done = (self.current_step >= self.max_steps)
        return s_next, r, done, {}


class CliffWalkingEnv:
    """
    Cliff Walking Environment (Sutton & Barto).
    Grid: height x width (default 4x10).
    Start: Bottom-left (height - 1, 0).
    Goal: Bottom-right (height - 1, width - 1).
    Cliff: Bottom row between start and goal.
    Actions: 0: Up, 1: Right, 2: Down, 3: Left.
    Rewards: -1 per step, -100 for falling into the cliff (resets to start).
    """
    def __init__(self, height: int = 4, width: int = 8, max_steps: int = 50, seed: Optional[int] = None):
        self.height = height
        self.width = width
        self.num_states = height * width
        self.num_actions = 4
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        
        self.start_pos = (height - 1, 0)
        self.goal_pos = (height - 1, width - 1)
        self.cliff = [(height - 1, c) for c in range(1, width - 1)]
        
        self.pos = self.start_pos
        self.current_step = 0

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.width + pos[1]

    def reset(self) -> int:
        self.pos = self.start_pos
        self.current_step = 0
        return self.pos_to_state(self.pos)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        self.current_step += 1
        r, c = self.pos
        
        # 0: Up, 1: Right, 2: Down, 3: Left
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            c = min(self.width - 1, c + 1)
        elif action == 2:
            r = min(self.height - 1, r + 1)
        elif action == 3:
            c = max(0, c - 1)
            
        new_pos = (r, c)
        
        if new_pos in self.cliff:
            # Fell into cliff!
            self.pos = self.start_pos
            reward = -100.0
            done = (self.current_step >= self.max_steps)
        elif new_pos == self.goal_pos:
            # Goal reached!
            self.pos = new_pos
            reward = 10.0
            done = True
        else:
            self.pos = new_pos
            reward = -1.0
            done = (self.current_step >= self.max_steps)
            
        return self.pos_to_state(self.pos), reward, done, {}


class DeepSeaTabularEnv:
    """
    Canonical Deep Sea Environment (Osband et al., 2018 / 2019).
    State space: Discrete grid of size N x N (total states = N^2).
    State index: s = row * N + col.
    Start state: (0, 0) -> s = 0.
    Actions: 0 and 1.
    If action_mapping is enabled, each row has a randomly permuted mapping where
    one action moves right (towards treasure) and the other falls left.
    Step reward: -0.01 / N for moving right, 0.0 for falling left.
    Terminal reward: +1.0 only at (N-1, N-1).
    Probability of random walk reaching the chest is 2^{-N}.
    """
    def __init__(
        self,
        n: int = 10,
        action_mapping: bool = True,
        seed: Optional[int] = None
    ):
        self.n = n
        self.num_states = n * n
        self.num_actions = 2
        self.action_mapping = action_mapping
        self.rng = np.random.default_rng(seed)
        
        # Secret optimal action per row: 1 if unpermuted, else random choice
        if action_mapping:
            self.optimal_actions = self.rng.integers(0, 2, size=n)
        else:
            self.optimal_actions = np.ones(n, dtype=int)
            
        self.row = 0
        self.col = 0

    def reset(self) -> int:
        self.row = 0
        self.col = 0
        return self.row * self.n + self.col

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        # Determine if action is moving right (optimal) or falling left
        is_right = (action == self.optimal_actions[self.row])
        
        if is_right:
            self.col = min(self.n - 1, self.col + 1)
            step_reward = -0.01 / self.n
        else:
            self.col = max(0, self.col - 1)
            step_reward = 0.0
            
        self.row += 1
        done = (self.row >= self.n)
        
        terminal_reward = 0.0
        if done and self.col == self.n - 1:
            terminal_reward = 1.0  # Treasure chest reached!
            
        reward = step_reward + terminal_reward
        s_next = (self.row * self.n + self.col) if not done else 0
        return s_next, reward, done, {"treasure_reached": (done and self.col == self.n - 1)}

