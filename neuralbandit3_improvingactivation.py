from neuralbandit3_original import *

def create_network(input_size, hidden_size, activation):
    # feed-forward neural network with one layer
    # input -> hidden layer -> one reward prediction

    # changeable activation function, relu, tanh, leaky_relu. still needs to accept sigmoid (uses sigmoid as default)

    if activation == "relu":
        net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
    elif activation == "tanh":
        net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
    elif activation == "leaky_relu":
        net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
    else:
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

def initialize_networks(input_size, num_actions, configs, activation):
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
            net = create_network(input_size, config["hidden"],activation)

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
        n_features, K, configs, args.activation
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
    print(f"Activation function: {args.activation}")
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
        default="neuralbandit3_improving_metrics.csv"
    )
    parser.add_argument(
        "--activation",
        choices=["sigmoid", "relu", "tanh", "leaky_relu"],
        default="sigmoid"
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
    