## Imports

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import KBinsDiscretizer


## Reproducibility

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


## Dataset

def load_forest_cover(seed = 123): # default seed is 123
    bunch = fetch_covtype() # fetch_covtype returns the 54 standard CoverType predictors
    # Create 94 binary context features and 7 action labels

    raw_x = np.asarray(bunch.data, dtype=np.float32)
    y = np.asarray(bunch.target, dtype=np.int64) - 1

    continuous = raw_x[:, :10] # first 10 CoverType predictors are continuous
    binary = raw_x[:, 10:] # remaining 44 CoverType predictors are already binary indicator columns

    # reducing each continuous into five binary equal-frequency bins
    discretizer = KBinsDiscretizer(
        n_bins=5,
        encode="onehot-dense",
        strategy="quantile",
        subsample=None,
    )
    continuous_binned = discretizer.fit_transform(continuous).astype(np.float32)

    x = np.concatenate([continuous_binned, binary], axis=1).astype(np.float32)

    if x.shape[1] != 94:
        raise RuntimeError(
            f"Expected 94 context features, got {x.shape[1]}. "
            "Check the dataset version and preprocessing."
        )

    # standard rng generator
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    return x[order], y[order]


## Neural Networks

def get_model_configs(grid):
    # decides what set of hyperparameters to use
    # hidden = number of neurons in the network's hidden layer
    # lr = learning rate used by stochastic gradient descent
    
    # small is shorter list, for debugging
    if grid == "small":
        return [
            {"hidden": 5, "lr": 0.01},
            {"hidden": 5, "lr": 0.1},
            {"hidden": 25, "lr": 0.01},
            {"hidden": 25, "lr": 0.1},
        ]

    # full is the complete 5x3=15 combination grid
    return [
        {"hidden": h, "lr": lr}
        for h in [1,5,25,50,100]
        for lr in [0.01,0.1,1.0]
        # cycles through all combinations of h,lr
    ]

def create_network(input_size, hidden_size):
    # feed-forward neural network with one layer
    # input -> hidden layer -> one reward prediction

    # Sigmoid activation function, final output is between 0 and 1
    net = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Sigmoid(),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid(),
    )

    # initalizes every weight and bias to a random value between -0.5 and 0.5
    for parameter in net.parameters():
        nn.init.uniform_(parameter, -0.5,0.5) # fills with values drawn from a continuous uniform distribution
    return net

def initialize_networks(input_size, num_actions, configs):
    # will create a network for every action being considered
    # configs is determined by grid, is a list of dictionaries

    networks = []
    optimizers = []

    # config specifies the hidden-layer size and learning rate
    for config in configs:
        # both of these are list of lists
        networks_for_config = []
        optimizers_for_config = []

        for i in range(num_actions):
            # create neural network
            net = create_network(input_size, config["hidden"])

            # create SGD optimizer 
            optimizer = torch.optim.SGD(
                net.parameters(), lr=config["lr"]
            )

            # append to config networks and optimizers
            networks_for_config.append(net)
            optimizers_for_config.append(optimizer)

        # append to overall networks and optimizers
        networks.append(networks_for_config)
        optimizers.append(optimizers_for_config)

    return networks, optimizers


## EXP3 !!Important!!
# exp3 is how neural bandit also selects actions

def initialize_exp3(num_actions, num_models):
    # every selector starts with equal weights, so intially all model configurations are equally likely to be selected

    # ex: weights[k,m] is configuration m's weight for action k
    return np.ones((num_actions, num_models), dtype=np.float64) # shape=(number of actions, number of models)

def exp3_probabilities(action_weights, gamma_model):
    # converts one action's EXP3 weights into probabilities
    # gamma_model controls exploration

    shifted = action_weights - np.max(action_weights) # shifts array so max becomes 0, other values become negative
    exp_weights = np.exp(shifted) # converts unnormalized weights into raw non-negative scores
    base_probs = exp_weights / exp_weights.sum() # percentage of the whole

    M = len(action_weights)
    return (1.0 - gamma_model) * base_probs + gamma_model/M # softmax

def choose_model(action_weights, gamma_model, rng):
    # samples one neural network configuration for an action

    probs = exp3_probabilities(action_weights, gamma_model)
    model_index = int(rng.choice(len(probs), p=probs)) # picks a random neural network
    return model_index, float(probs[model_index])

def update_exp3(action_weights, model_index, model_probability, reward, gamma_model):
    # updates the exp3 selector after observing reward (1 or 0)

    M = len(action_weights)
    p = max(float(model_probability), 1e-12)
    exponent = gamma_model * float(reward) / (p*M)
    action_weights[model_index] *= np.exp(exponent)

    action_weights /= np.max(action_weights) # scales all weights by the same constant, preserve probabilities


## Action selection/Neural Network learning

def choose_action(networks, model_for_action, context, gamma, rng):
    # for action k, model_for_action[k] tells us which network configuration exp3 selected

    k_total = len(model_for_action)
    x = torch.as_tensor(
        context, dtype=torch.float32
    ).reshape(1,-1)

    # each action receives a predicted reward score from its selected network
    scores = np.zeros(k_total, dtype=np.float64)
    with torch.no_grad():
        for k in range(k_total):
            m = model_for_action[k]
            scores[k] = networks[m][k](x).item()

    greedy_action = int(np.argmax(scores)) # greedy action has largest predicted reward

    action_probs = np.full(k_total, gamma/k_total, dtype=np.float64)
    action_probs[greedy_action] += 1.0 - gamma # probability(action k) = (1-gamma) * I[k=greedy_action] + gamma/K
    action = int(rng.choice(k_total, p=action_probs)) # usually chooses greedy_action, sometimes doesn't to explore for better reward

    # greedy_action comes from just exploitation
    # action comes from balance of exploration and exploitation
    return action, greedy_action, action_probs

def update_selected_action_networks(networks, optimizers, action, context, reward, action_probability, max_importance_weight):
    # updates all candidate networks for the action that was undergone
    # no update for unplayed actions

    x = torch.as_tensor(
            context, dtype=torch.float32
    ).reshape(1,-1)
    target = torch.tensor([[float(reward)]], dtype=torch.float32)

    importance = 1.0 / max(float(action_probability), 1e-12) # inverse-propensity importance weighting
    importance = min(importance, max_importance_weight) # max_importance_weight is a safety cap

    # actual update step
    # updates every condidate's configuration network for actions that were played
    for m in range(len(networks)):
        net = networks[m][action]
        optimizer = optimizers[m][action]
        optimizer.zero_grad()
        prediction = net(x)
        loss = torch.mean((prediction-target)**2)
        (importance*loss).backward() #
        optimizer.step()


## simulation

def drifted_label(label, step, drift_every):
    # This function creates artificial concept drift for experiments.
    # If drift_every is 0, labels never change. Otherwise, every drift_every
    # iterations all class labels rotate forward by one (wrapping after 7).
    # This is the cyclic class rotation used in the paper's drift experiment.
    if drift_every <= 0:
        return int(label)

    shifts = step // drift_every

    return int((label + shifts) % 7)

def run(args):
    # Main experiment procedure: prepare data, initialize the algorithm,
    # then repeat the contextual-bandit decision/update cycle.
    set_seeds(args.seed)

    print("Loading Forest Cover Type dataset...")
    X, y = load_forest_cover(args.seed)
    n_rows, n_features = X.shape
    K = 7
    configs = get_model_configs(args.models)
    M = len(configs)

    print(f"Rows: {n_rows:,} | Features: {n_features} | Actions: {K}")
    print(f"Model configurations: {M}")

    # Set up the K * M neural networks and their optimizers.
    networks, optimizers = initialize_networks(
        n_features, K, configs
    )

    # Set up K separate EXP3 selectors: one model selector for each action.
    exp3_weights = initialize_exp3(K, M)
    rng = np.random.default_rng(args.seed)

    cumulative_reward = 0.0
    cumulative_mistakes = 0.0
    reward_history = []
    start_time = time.time()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "step",
        "reward",
        "rolling_reward",
        "cumulative_reward",
        "cumulative_regret",
        "selected_action",
        "true_action",
        "greedy_action",
        "drift_event",
        "drift_number",
        "steps_since_drift",
        "elapsed_seconds"
    ]

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        for step in range(args.steps):
            # The dataset is treated as a stream and replayed from the start
            # when we reach its final row. This lets experiments run for more
            # iterations than there are distinct observations.
            
            # Convert zero-based step into human-readable iteration
            iteration = step + 1
            i = step % n_rows
            context = X[i]
            
            # Apply concept drift
            true_action = drifted_label(
                y[i],
                iteration,
                args.drift_every
            )

            # Track concept drift information
            if args.drift_every > 0:
        
                drift_event = (
                    iteration % args.drift_every == 0
                )
        
                drift_number = iteration // args.drift_every
        
                if iteration < args.drift_every:
                    steps_since_drift = None
                else:
                    steps_since_drift = iteration % args.drift_every
        
            else:
                drift_event = False
                drift_number = 0
                steps_since_drift = None

            # For every possible action, use that action's EXP3 selector to
            # sample a neural-network configuration. NeuralBandit3 therefore
            # allows different actions to use different configurations.
            model_for_action = []
            model_probability_for_action = []
            for k in range(K):
                m, p_m = choose_model(
                    exp3_weights[k], args.gamma_model, rng
                )
                model_for_action.append(m)
                model_probability_for_action.append(p_m)

            # Score all actions with their selected networks, then sample
            # the action using the paper's action-exploration distribution.
            action, greedy_action, action_probs = choose_action(
                networks, model_for_action, context, args.gamma, rng
            )
            action_probability = float(action_probs[action])

            # Bandit feedback: the learner observes only the reward for the
            # action it selected. Here the experiment can calculate that reward
            # because the dataset's true label is available to the simulator.
            # Correct action -> reward 1; incorrect action -> reward 0.
            reward = 1.0 if action == true_action else 0.0

            # Train every candidate network associated with the selected action.
            update_selected_action_networks(
                networks=networks,
                optimizers=optimizers,
                action=action,
                context=context,
                reward=reward,
                action_probability=action_probability,
                max_importance_weight=args.max_importance_weight
            )

                        # Only the selected action's EXP3 selector receives feedback,
            # because rewards for all other actions remain unknown.
            # The chosen model is the configuration that supplied the score
            # for the action that was actually selected.
            chosen_model = model_for_action[action]
            update_exp3(
                exp3_weights[action],
                chosen_model,
                model_probability_for_action[action],
                reward,
                args.gamma_model,
            )

            # Track total reward and total mistakes. Since reward is binary,
            # each incorrect choice adds one to cumulative_mistakes.
            cumulative_reward += reward
            cumulative_mistakes += 1.0 - reward
            reward_history.append(reward)
            if len(reward_history) > args.rolling_window:
                reward_history.pop(0)

            if (
                iteration % args.log_every == 0
                or iteration == 1
                or iteration == args.steps
                or drift_event
            ):
                rolling = float(np.mean(reward_history))
                elapsed = time.time() - start_time
                print(
                    f"Step {step + 1:,}/{args.steps:,} | "
                    f"rolling reward={rolling:.4f} | "
                    f"total reward={cumulative_reward:.1f} | "
                    f"mistakes={cumulative_mistakes:.1f} | "
                    f"elapsed={elapsed:.1f}s"
                )
                writer.writerow({
                    "step": iteration,
                    "reward": reward,
                    "rolling_reward": rolling,
                    "cumulative_reward": cumulative_reward,
                    "cumulative_regret": cumulative_mistakes,
                    "selected_action": action,
                    "true_action": true_action,
                    "greedy_action": greedy_action,
                    "drift_event": drift_event,
                    "drift_number": drift_number,
                    "steps_since_drift": steps_since_drift,
                    "elapsed_seconds": elapsed
                })
                csv_file.flush()

    print(f"\nMetrics saved to: {output.resolve()}")
    print(f"Total reward: {cumulative_reward:.1f}")
    print(f"Total mistakes: {cumulative_mistakes:.1f}")
    print(f"Final rolling reward: {np.mean(reward_history):.4f}")


def parse_args():
    # Read settings supplied when the script is launched from a terminal.
    # For example, "--models full" makes args.models equal "full".
    parser = argparse.ArgumentParser(
        description="NeuralBandit3 without user-defined classes"
    )
    parser.add_argument("--steps", type=int, default=2000)
    # Choose the hyperparameter grid: "small" for quick tests or "full"
    # for all 15 configurations described in the paper.
    parser.add_argument("--models", choices=["small", "full"], default="small")
    # gamma controls exploration over the seven actions.
    # gamma-model controls exploration over neural-network configurations
    # within each action's EXP3 selector.
    parser.add_argument("--gamma", type=float, default=0.005)
    parser.add_argument("--gamma-model", type=float, default=0.1)
    # Set to 0 for stationary labels. A positive value rotates class labels
    # after every specified number of iterations.
    parser.add_argument("--drift-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--rolling-window", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    # "auto" uses a CUDA-capable GPU if available; otherwise it uses the CPU.
    # Upper limit on the inverse-probability training weight. This prevents
    # rare action selections from producing excessively large gradient steps.
    parser.add_argument("--max-importance-weight", type=float, default=100.0)
    parser.add_argument(
        "--output", type=str,
        default="neuralbandit3_original_metrics.csv"
    )
    args = parser.parse_args()

    if args.steps <= 0 or args.log_every <= 0 or args.rolling_window <= 0:
        parser.error("--steps, --log-every, and --rolling-window must be positive")
    if not 0 <= args.gamma <= 0.5:
        parser.error("--gamma must be between 0 and 0.5")
    if not 0 <= args.gamma_model <= 0.5:
        parser.error("--gamma-model must be between 0 and 0.5")
    if args.drift_every < 0 or args.max_importance_weight <= 0:
        parser.error("--drift-every must be nonnegative and importance cap positive")
    return args


if __name__ == "__main__":
    # This block runs only when this file is executed directly.
    # First parse the user's command-line settings, then start the experiment.
    run(parse_args())