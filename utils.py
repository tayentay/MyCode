import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=bool, default=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=bool, default=False,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--project-name", type=str, default="uav-drl",
                        help="the wandb's project name")
    parser.add_argument("--out-subdir", type=str, default="runs",
                        help="the subdirectory for output files")
    
    # Training parameters
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=64,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-updates", type=int, default=1000,
                        help="total timesteps of the experiments")
    parser.add_argument("--anneal-lr", type=bool, default=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=bool, default=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=bool, default=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=bool, default=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="batch size (computed in code if not provided)")
    parser.add_argument("--minibatch-size", type=int, default=None,
                        help="minibatch size (computed in code if not provided)")
    
    # Model saving and logging
    parser.add_argument("--save-model", type=bool, default=True,
                        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="interval to save model checkpoints")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="interval to log metrics")
    parser.add_argument("--load-model-path", type=str, default=None,
                        help="path to load a pretrained model")
    parser.add_argument("--eval-mode", type=bool, default=False,
                        help="if toggled, run in evaluation mode (no training)")
    
    args = parser.parse_args()
    
    # Compute batch_size and minibatch_size if not provided
    if args.batch_size is None:
        args.batch_size = int(args.num_steps)
    if args.minibatch_size is None:
        args.minibatch_size = int(args.batch_size // 4)
    
    return args
