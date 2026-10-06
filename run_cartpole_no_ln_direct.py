#!/usr/bin/env python3
"""Run Cart-Pole Swing-Up Pure TS without LayerNorm (use_layer_norm=False)
Directly comparable to results_cartpole_pure_ts/cartpole_swingup_dp_dqn_deepsea_construct_gaussian_pure_ts_a15.0_s*.json
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import concurrent.futures
import json
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
torch.set_num_threads(1)

from src.dp_dqn.agent import DPDQNAgent
from src.dp_dqn.config import DPDQNConfig
from src.dp_dqn.environments import make_env


def run_cartpole_no_ln_single(seed: int, alpha: float = 15.0, max_episodes: int = 1500, out_dir: str = "./results_no_ln") -> Dict[str, Any]:
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"cartpole_no_ln_gaussian_pure_ts_a{alpha}_s{seed}.json")
    ckpt_path = os.path.join(out_dir, f"cartpole_no_ln_gaussian_pure_ts_a{alpha}_s{seed}_ckpt.json")

    t0 = time.time()
    env = make_env("cartpole_swingup", seed=seed)
    state_dim = env.state_dim
    action_dim = env.action_dim

    cfg = DPDQNConfig(
        env_name="cartpole_swingup",
        state_dim=state_dim,
        action_dim=action_dim,
        seed=seed,
        alpha=alpha,
        batch_size=64,
        candidate_batch_size=256,
        base_measure="gaussian",
        prior_reward_mean=0.5,
        prior_reward_std=0.5,
        hidden_dim=64,
        num_layers=2,
        use_layer_norm=False,  # STRICTLY NO LAYERNORM
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        sgd_period=2,
        episodic_sgd=False,
        buffer_capacity=1000000,
        target_warmstart=True,
        warmstart_steps=2,
        warmstart_lr_scale=1.0,
        sample_once_per_episode=True,  # 100% Pure TS
        num_episodes=max_episodes,
        max_episode_steps=1000,
        sampler_type="vashishtha_maillard",
        vm_prior_multiplier=10.0,
        one_living_network=True,
        dp_sampled_target=False,
        use_td_info_gain_decay=True,
        td_info_scale=1.0,
        verbose=False,
    )

    agent = DPDQNAgent(cfg)

    returns: List[float] = []
    uprights: List[int] = []
    first_swingup: Optional[int] = None
    solved_episode: Optional[int] = None
    cum_upright = 0
    cum_regret = 0.0

    print(f"--> [START CARTPOLE NO-LN] Seed={seed} Alpha={alpha} (use_layer_norm=False)", flush=True)

    for ep in range(1, max_episodes + 1):
        agent.reset_episode()
        s = env.reset()
        ep_ret = 0.0
        ep_upr = 0
        done = False
        steps = 0

        while not done:
            a = agent.act(s)
            sn, r, done, info = env.step(a)
            agent.step(s, a, float(r), sn, done)
            s = sn
            ep_ret += float(r)
            if info.get("is_upright", False):
                ep_upr += 1
            steps += 1

        agent.end_episode(steps)
        returns.append(ep_ret)
        uprights.append(ep_upr)
        cum_upright += ep_upr
        reg = max(0.0, 1000.0 - ep_ret)
        cum_regret += reg

        if ep_upr > 100 and first_swingup is None:
            first_swingup = ep
            print(f"*** [CARTPOLE NO-LN] FIRST SWING-UP Seed={seed} at Ep {ep}! (Upr={ep_upr}, Ret={ep_ret:.1f}, {time.time()-t0:.1f}s)", flush=True)

        if ep >= 50:
            rec_ret = float(np.mean(returns[-50:]))
            rec_upr = float(np.mean(uprights[-50:]))
            if rec_ret >= 500.0 or rec_upr >= 500.0:
                if solved_episode is None:
                    solved_episode = ep
                    print(f"*** [CARTPOLE NO-LN] SOLVED Seed={seed} at Ep {ep}! (50-MA Ret={rec_ret:.1f}, {time.time()-t0:.1f}s)", flush=True)

        if ep % 50 == 0:
            rec_ret = float(np.mean(returns[-min(ep, 50):]))
            rec_upr = float(np.mean(uprights[-min(ep, 50):]))
            print(f"[CARTPOLE NO-LN] Seed={seed} Ep={ep:4d}/{max_episodes} | Ret(50): {rec_ret:6.1f} | Upr(50): {rec_upr:5.1f} | Disc: {first_swingup} | Solv: {solved_episode} | Time: {time.time()-t0:.1f}s", flush=True)
            
            # Save checkpoint
            ckpt_data = {
                "benchmark": "cartpole_no_layernorm",
                "seed": seed,
                "alpha": alpha,
                "use_layer_norm": False,
                "first_swingup": first_swingup,
                "solved_episode": solved_episode,
                "current_episode": ep,
                "returns": returns,
                "upright_steps": uprights,
                "cum_regret": float(cum_regret),
                "elapsed_seconds": time.time() - t0,
            }
            with open(ckpt_path, "w") as fp:
                json.dump(ckpt_data, fp)

    elapsed = time.time() - t0
    res = {
        "benchmark": "cartpole_no_layernorm",
        "seed": seed,
        "alpha": alpha,
        "use_layer_norm": False,
        "first_swingup": first_swingup,
        "solved_episode": solved_episode,
        "peak_return": float(np.max(returns)),
        "final_50_return": float(np.mean(returns[-50:])),
        "total_cum_upright": cum_upright,
        "total_episodes": len(returns),
        "returns": returns,
        "upright_steps": uprights,
        "cum_regret": float(cum_regret),
        "elapsed_seconds": elapsed,
        "completed": True,
    }
    with open(fpath, "w") as fp:
        json.dump(res, fp)
    print(f"--> [FINISH CARTPOLE NO-LN] Seed={seed} -> Solved={solved_episode} PeakRet={res['peak_return']:.1f} in {elapsed:.1f}s", flush=True)
    return res


def main():
    seeds = [42, 43, 44]
    print("==========================================================")
    print(" Running Cart-Pole Swing-Up No-LayerNorm (Pure TS, a=15.0)")
    print(f" Seeds: {seeds}")
    print("==========================================================")
    t_start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_cartpole_no_ln_single, s): s for s in seeds}
        for future in concurrent.futures.as_completed(futures):
            s = futures[future]
            try:
                res = future.result()
                print(f"[COMPLETED] CartPole Seed {s} finished: Solved={res['solved_episode']}, PeakRet={res['peak_return']:.1f}")
            except Exception as e:
                import traceback
                print(f"[ERROR] CartPole Seed {s} failed: {e}\n{traceback.format_exc()}")
    print(f"All CartPole runs finished in {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
