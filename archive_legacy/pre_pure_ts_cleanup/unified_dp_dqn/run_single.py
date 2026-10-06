"""CLI runner for a single experiment using the unified DP-DQN framework."""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Ensure parent directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from unified_dp_dqn import (
    DPDQNAgent,
    DPDQNConfig,
    make_env,
    BootDQNRPAgent,
    BayesianDeepQNetworkAgent,
    StandardDQNAgent
)


def create_agent(algo: str, env, seed: int, alpha: float = 3.0, base_measure: str = None, sampler: str = None):
    state_dim = env.state_dim
    action_dim = env.action_dim
    is_deep_sea = hasattr(env, "size") and (state_dim == env.size * env.size)
    default_bm = base_measure if base_measure else ("deep_sea" if is_deep_sea else "haar")
    chosen_sampler = sampler if sampler else "vashishtha_maillard"

    if algo == "dp_dqn_haar":
        bm = "deep_sea" if is_deep_sea else "haar"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=alpha,
            base_measure_type=bm,
            haar_angle=(not is_deep_sea),
            seed=seed,
            buffer_capacity=3000000
        )
        return DPDQNAgent(cfg)
    elif algo in ("dp_dqn_non_haar", "dp_dqn_dag"):
        bm = "deep_sea" if is_deep_sea else "uniform"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=alpha,
            base_measure_type=bm,
            haar_angle=False,
            seed=seed,
            buffer_capacity=3000000
        )
        return DPDQNAgent(cfg)
    elif algo in ("dp_dqn_dirichlet", "dp_dqn_dirichlet_wmin"):
        bm = "deep_sea" if is_deep_sea else "haar"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=alpha,
            base_measure_type=bm,
            haar_angle=(not is_deep_sea),
            sampler_type="dirichlet",
            w_min=0.05,
            seed=seed,
            buffer_capacity=3000000
        )
        return DPDQNAgent(cfg)
    elif algo in ("dp_dqn_stick", "dp_dqn_stick_wmin"):
        bm = "deep_sea" if is_deep_sea else "haar"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=alpha,
            base_measure_type=bm,
            haar_angle=(not is_deep_sea),
            sampler_type="vashishtha_maillard",
            w_min=0.05,
            seed=seed,
            buffer_capacity=3000000
        )
        return DPDQNAgent(cfg)
    elif algo == "dp_dqn_alpha_small":
        bm = "deep_sea" if is_deep_sea else "haar"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            alpha=1e-10,
            base_measure_type=bm,
            haar_angle=(not is_deep_sea),
            seed=seed,
            buffer_capacity=1000000
        )
        return DPDQNAgent(cfg)
    elif algo == "dp_dqn_vm_ln_64":
        bm = "deep_sea" if is_deep_sea else "haar"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_layer_norm=True,
            alpha=alpha,
            base_measure_type=bm,
            haar_angle=(not is_deep_sea),
            sampler_type="vashishtha_maillard",
            w_min=0.05,
            seed=seed,
            buffer_capacity=3000000,
            warmstart_steps=2,
            dp_online_sgd=False,
            tau=0.05,
        )
        return DPDQNAgent(cfg)
    elif algo == "dp_dqn_vm_noln_64":
        bm = "deep_sea" if is_deep_sea else "haar"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_layer_norm=False,
            alpha=alpha,
            base_measure_type=bm,
            haar_angle=(not is_deep_sea),
            sampler_type="vashishtha_maillard",
            w_min=0.05,
            seed=seed,
            buffer_capacity=3000000,
            warmstart_steps=2,
            dp_online_sgd=False,
            tau=0.05,
        )
        return DPDQNAgent(cfg)
    elif algo == "dp_dqn_dir_ln_64":
        bm = "deep_sea" if is_deep_sea else "haar"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_layer_norm=True,
            alpha=alpha,
            base_measure_type=bm,
            haar_angle=(not is_deep_sea),
            sampler_type="dirichlet",
            w_min=0.05,
            seed=seed,
            buffer_capacity=3000000,
            warmstart_steps=2,
            dp_online_sgd=False,
            tau=0.05,
        )
        return DPDQNAgent(cfg)
    elif algo == "dp_dqn_dir_noln_64":
        bm = "deep_sea" if is_deep_sea else "haar"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_layer_norm=False,
            alpha=alpha,
            base_measure_type=bm,
            haar_angle=(not is_deep_sea),
            sampler_type="dirichlet",
            w_min=0.05,
            seed=seed,
            buffer_capacity=3000000,
            warmstart_steps=2,
            dp_online_sgd=False,
            tau=0.05,
        )
        return DPDQNAgent(cfg)
    elif algo in ("dp_dqn_online_dp", "dp_dqn_dir_online_dp"):
        bm = "deep_sea" if is_deep_sea else "haar"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_layer_norm=False,
            alpha=alpha,
            base_measure_type=bm,
            haar_angle=(not is_deep_sea),
            sampler_type="dirichlet",
            w_min=0.05,
            seed=seed,
            buffer_capacity=3000000,
            warmstart_steps=2,
            dp_online_sgd=True,
            tau=0.05,
        )
        return DPDQNAgent(cfg)
    elif algo == "dp_dqn_vm_online_dp":
        bm = "deep_sea" if is_deep_sea else "haar"
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_layer_norm=False,
            alpha=alpha,
            base_measure_type=bm,
            haar_angle=(not is_deep_sea),
            sampler_type="vashishtha_maillard",
            w_min=0.05,
            seed=seed,
            buffer_capacity=3000000,
            warmstart_steps=2,
            dp_online_sgd=True,
            tau=0.05,
        )
    elif algo in ("dp_dqn_deepsea_construct", "dp_dqn_construct"):
        cfg = DPDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_layer_norm=True,
            alpha=alpha,
            base_measure_type=default_bm,
            haar_angle=False,
            sampler_type=chosen_sampler,
            w_min=0.05,
            seed=seed,
            buffer_capacity=1000000,
            warmstart_steps=2,
            dp_online_sgd=False,  # EXACTLY non-DP online gradient steps!
            tau=0.05,            # EXACTLY Polyak target!
        )
        return DPDQNAgent(cfg)
    elif algo == "boot_dqn":
        return BootDQNRPAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_models=20,
            hidden_dim=50,
            prior_scale=1.0,
            seed=seed,
            capacity=1000000
        )
    elif algo == "bdqn":
        return BayesianDeepQNetworkAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=50,
            prior_variance=1.0,
            noise_variance=0.1,
            seed=seed,
            capacity=1000000
        )
    elif algo == "vanilla_dqn":
        return StandardDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=50,
            epsilon_start=1.0,
            epsilon_end=0.01,
            anneal_episodes=500,
            seed=seed,
            capacity=1000000
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def main():
    parser = argparse.ArgumentParser(description="Run single unified experiment")
    parser.add_argument("--env", type=str, default="cartpole_swingup",
                        help="Environment name: cartpole_swingup, deep_sea, or gym id")
    parser.add_argument("--size", type=int, default=10,
                        help="Size parameter (for deep_sea N)")
    parser.add_argument("--algo", type=str, default="dp_dqn_deepsea_construct",
                        help="Algorithm to run")
    parser.add_argument("--base_measure", type=str, default=None,
                        help="Base measure type: empirical, gaussian, haar, uniform")
    parser.add_argument("--sampler", type=str, default="vashishtha_maillard",
                        help="Sampler type: vashishtha_maillard or dirichlet")
    parser.add_argument("--alpha", type=float, default=3.0,
                        help="Dirichlet Process concentration parameter alpha")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--episodes", type=int, default=2500,
                        help="Total episodes")
    parser.add_argument("--warmstart_steps", type=int, default=None,
                        help="Number of episodic warmstart steps")
    parser.add_argument("--dp_online_sgd", action="store_true",
                        help="Use DP posterior sampler for online SGD during episodes")
    parser.add_argument("--sample_once_per_episode", action="store_true",
                        help="Sample DP posterior once per episode (Pure TS)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag for output file naming")
    parser.add_argument("--out_dir", type=str, default="./results_unified",
                        help="Directory to store JSON results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_file = os.path.join(args.out_dir, f"{args.env}_{args.algo}{suffix}_s{args.seed}.json")
    ckpt_file = os.path.join(args.out_dir, f"{args.env}_{args.algo}{suffix}_s{args.seed}_ckpt.json")

    env = make_env(args.env, seed=args.seed, size=args.size)
    agent = create_agent(args.algo, env, seed=args.seed, alpha=args.alpha, base_measure=args.base_measure, sampler=args.sampler)
    if args.warmstart_steps is not None and hasattr(agent, "config"):
        agent.config.warmstart_steps = int(args.warmstart_steps)
    if args.dp_online_sgd and hasattr(agent, "config"):
        agent.config.dp_online_sgd = True
    if args.sample_once_per_episode and hasattr(agent, "config"):
        agent.config.sample_once_per_episode = True



    returns = []
    uprights = []
    regrets = []
    cumulative_regrets = []
    cum_regret = 0.0
    first_discovery = None
    solved_episode = None
    recent_solved_counter = 0

    r_star = 850.0 if "cartpole" in args.env else (1.0 - 0.01)

    t0 = time.time()
    for ep in range(1, args.episodes + 1):
        agent.reset_episode()
        state = env.reset()
        done = False
        ep_ret = 0.0
        upr = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)

            ep_ret += reward
            if info.get("is_upright", False):
                upr += 1
            state = next_state

        inst_regret = r_star - ep_ret
        cum_regret += inst_regret

        returns.append(float(ep_ret))
        uprights.append(int(upr))
        regrets.append(float(inst_regret))
        cumulative_regrets.append(float(cum_regret))

        if ep_ret > 0.5:
            if first_discovery is None:
                first_discovery = ep
                print(f"[{args.env} | {args.algo} | s{args.seed}] *** FIRST DISCOVERY at Episode {ep}! ***")
            recent_solved_counter += 1
            if recent_solved_counter >= 10 and solved_episode is None:
                solved_episode = ep - 9
                print(f"[{args.env} | {args.algo} | s{args.seed}] *** SOLVED ENVIRONMENT at Episode {solved_episode}! ***")
        else:
            recent_solved_counter = 0

        if ep % 50 == 0 or ep == args.episodes:
            elapsed = time.time() - t0
            ret_50 = np.mean(returns[-50:])
            upr_50 = np.mean(uprights[-50:])
            print(f"[{args.env} | {args.algo} | s{args.seed}] Ep {ep:4d}/{args.episodes} | "
                  f"Ret(50): {ret_50:6.2f} | Upr(50): {upr_50:5.1f} | "
                  f"CumReg: {cum_regret/1e3:6.1f}k | Time: {elapsed:5.1f}s")

            ckpt = {
                "env": args.env,
                "algo": args.algo,
                "seed": args.seed,
                "alpha": args.alpha,
                "episodes": ep,
                "first_discovery": first_discovery,
                "solved_episode": solved_episode,
                "returns": returns,
                "upright_steps": uprights,
                "regrets": regrets,
                "cumulative_regrets": cumulative_regrets,
                "elapsed_sec": elapsed
            }
            with open(ckpt_file, "w") as f:
                json.dump(ckpt, f)

    with open(out_file, "w") as f:
        json.dump(ckpt, f, indent=2)
    print(f"Finished {args.algo} on {args.env} (seed {args.seed}) in {time.time() - t0:.1f}s -> {out_file}")


if __name__ == "__main__":
    main()
