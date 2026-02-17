import sys
from ns import ns
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import shutil, os
from env import EnvForDRL
from config import Config
from utils import parse_args

# ====== Self-Predictive Module (decoupled import) ======
from self_predictive import SelfPredictiveModule

# ====== Preference Estimator ======
from reward_m import PreferenceEstimator

# ====== GradNorm-Lite ======
from gradnorm import GradNormBalancer

# torch.cuda.set_device(0)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

g_sim = None

def scheduling_callback(args):
    global g_sim
    if g_sim:
        g_sim.do_scheduling()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks.type(torch.BoolTensor).to(device)
        super(CategoricalMasked, self).__init__(
            probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p,
                              torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    """
    PPO Agent with optional self-predictive encoder.
    """
    def __init__(self, env, sp_encoder=None):
        super(Agent, self).__init__()
        self.env = env
        self.sp_encoder = sp_encoder

        if self.sp_encoder is not None:
            input_dim = self.sp_encoder.latent_dim
        else:
            input_dim = self.env.single_observation_space

        self.network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.LeakyReLU()
        )
        self.nvec = self.env.nvec
        self.actor = layer_init(nn.Linear(128, self.nvec.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def _get_features(self, x):
        if self.sp_encoder is not None:
            z = self.sp_encoder(x)
            return self.network(z)
        else:
            return self.network(x)

    def get_value(self, x):
        return self.critic(self._get_features(x))

    def get_action_and_value(self, x, action_mask=None, action=None, deterministic=False):
        hidden = self._get_features(x)
        logits = self.actor(hidden)
        if action_mask is None:
            action_mask = torch.ones_like(logits).to(device)
        split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
        split_action_masks = torch.split(action_mask, self.nvec.tolist(), dim=1)
        multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_action_masks)]
        if action is None:
            if deterministic:
                action = torch.stack([torch.argmax(categorical.probs, dim=-1) for categorical in multi_categoricals])
            else:
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action.T, logprob.sum(0), entropy.sum(0), self.critic(hidden), action_mask


if not hasattr(ns.cppyy.gbl, 'pythonMakeEvent'):
    ns.cppyy.cppdef("""
    EventImpl* pythonMakeEvent(void (*f)(std::vector<std::string>), std::vector<std::string> l)
    {
        return MakeEvent(f, l);
    }
    """
    )

class Main:

    def __init__(self, args, config: Config, env):
        self.args = args
        self.config = config
        self.env = env
        self.step = 0
        # TRY NOT TO MODIFY: seeding
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        # ====== Self-Predictive Module Initialization ======
        self.use_self_predictive = config.use_self_predictive
        if self.use_self_predictive:
            sp_latent_dim = config.sp_latent_dim
            sp_hidden_dim = config.sp_hidden_dim
            sp_model_hidden_dim = config.sp_model_hidden_dim
            sp_model_layers = config.sp_model_layers
            sp_tau = config.sp_tau
            self.sp_coef = config.sp_coef
            self.sp_lr = config.sp_lr
            self.total_action_dim = int(env.nvec.sum())

            self.sp_module = SelfPredictiveModule(
                obs_dim=env.single_observation_space,
                action_dim=self.total_action_dim,
                latent_dim=sp_latent_dim,
                hidden_dim=sp_hidden_dim,
                model_hidden_dim=sp_model_hidden_dim,
                model_num_layers=sp_model_layers,
                tau=sp_tau,
                attn_num_heads=getattr(config, 'sp_attn_num_heads', 4),
                attn_num_layers=getattr(config, 'sp_attn_num_layers', 1),
                attn_dropout=getattr(config, 'sp_attn_dropout', 0.0),
                device=device,
            )
            self.agent = Agent(env, sp_encoder=self.sp_module.encoder).to(device)

            self.optimizer = optim.Adam(
                list(self.agent.parameters()) + list(self.sp_module.encoder.parameters()),
                lr=args.learning_rate, eps=1e-5
            )
            self.sp_model_optimizer = optim.Adam(
                self.sp_module.get_transition_model_parameters(),
                lr=self.sp_lr, eps=1e-5
            )

            # ====== GradNorm-Lite Initialization ======
            gradnorm_alpha = getattr(config, 'gradnorm_alpha', 1.5)
            gradnorm_lr = getattr(config, 'gradnorm_lr', 0.025)
            self.gradnorm = GradNormBalancer(
                num_tasks=2,
                alpha=gradnorm_alpha,
                lr=gradnorm_lr,
                device=device,
            )
            self.shared_params_for_gradnorm = list(self.sp_module.encoder.parameters())
        else:
            self.sp_module = None
            self.gradnorm = None
            self.agent = Agent(env, sp_encoder=None).to(device)
            self.optimizer = optim.Adam(
                self.agent.parameters(), lr=args.learning_rate, eps=1e-5
            )
            print("[Self-Predictive] Disabled, using standard PPO.")
        
        # ====== Preference Estimator Initialization ======
        self.use_preference_algo = getattr(config, 'enable_preference_algo', False)
        if self.use_preference_algo:
            self.preference_estimator = PreferenceEstimator(config, config.num_ues, config.num_uavs)
            self.preference_warmup_steps = getattr(config, 'preference_warmup_steps', 500)
            self.total_steps_taken = 0 # Track total steps across episodes
            self.preference_switched = False
            print(f"[Preference Algo] Enabled. Warmup={self.preference_warmup_steps}, min_buffer={self.preference_estimator.min_buffer_size}")
        else:
            self.preference_estimator = None

        # ALGO Logic: Storage setup
        self.obs = torch.zeros(
            (args.num_steps, env.single_observation_space)).to(device)
        self.actions = torch.zeros(
            (args.num_steps, env.single_action_space)).to(device)
        self.logprobs = torch.zeros(args.num_steps, 1).to(device)
        self.rewards = torch.zeros(args.num_steps, 1).to(device)
        self.dones = torch.zeros(args.num_steps, 1).to(device)
        self.values = torch.zeros(args.num_steps, 1).to(device)
        self.action_masks = torch.zeros((args.num_steps, env.sum_action)).to(device)
        self.next_obs = None
        self.next_done = None
        self.sum_inf = None
        self.metrics = None
        self.episode_duration = self.args.num_steps * self.config.time_slot_duration
        self.config.simulation_time = self.episode_duration

    def save_checkpoint(self, filepath):
        checkpoint = {
            'agent': self.agent.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'use_self_predictive': self.use_self_predictive,
        }
        if self.use_self_predictive:
            checkpoint['sp_module'] = self.sp_module.state_dict()
            checkpoint['sp_model_optimizer'] = self.sp_model_optimizer.state_dict()
            if self.gradnorm is not None:
                checkpoint['gradnorm'] = {
                    'weights': self.gradnorm.weights,
                    '_grad_norm_ema': self.gradnorm._grad_norm_ema,
                    '_loss_ema': self.gradnorm._loss_ema,
                    '_initial_losses': self.gradnorm._initial_losses
                }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}")
            return
            
        checkpoint = torch.load(filepath)
        self.agent.load_state_dict(checkpoint['agent'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.use_self_predictive and checkpoint.get('use_self_predictive', False):
            if 'sp_module' in checkpoint:
                self.sp_module.load_state_dict(checkpoint['sp_module'])
            if 'sp_model_optimizer' in checkpoint:
                self.sp_model_optimizer.load_state_dict(checkpoint['sp_model_optimizer'])
            if 'gradnorm' in checkpoint and self.gradnorm is not None:
                gn_state = checkpoint['gradnorm']
                self.gradnorm.weights = gn_state['weights'].to(self.gradnorm.device)
                if gn_state['_grad_norm_ema'] is not None:
                    self.gradnorm._grad_norm_ema = gn_state['_grad_norm_ema'].to(self.gradnorm.device)
                if gn_state['_loss_ema'] is not None:
                    self.gradnorm._loss_ema = gn_state['_loss_ema'].to(self.gradnorm.device)
                if gn_state['_initial_losses'] is not None:
                    self.gradnorm._initial_losses = gn_state['_initial_losses'].to(self.gradnorm.device)
        
        print(f"Model loaded from {filepath}")

    def _actions_to_onehot(self, actions: torch.Tensor) -> torch.Tensor:
        nvec = self.env.nvec
        parts = []
        for i, n in enumerate(nvec):
            a = actions[:, i].long()
            onehot = torch.zeros(a.shape[0], int(n), device=device)
            onehot.scatter_(1, a.unsqueeze(1), 1.0)
            parts.append(onehot)
        return torch.cat(parts, dim=-1)

    def _compute_self_predictive_loss(self, mb_obs, mb_actions):
        if not self.use_self_predictive or self.sp_module is None:
            return torch.tensor(0.0, device=device)

        valid_steps = min(self.step, self.args.num_steps - 1)
        if valid_steps < 2:
            return torch.tensor(0.0, device=device)

        obs_t = self.obs[:valid_steps]
        obs_tp1 = self.obs[1:valid_steps + 1]
        act_t = self.actions[:valid_steps]

        act_onehot = self._actions_to_onehot(act_t)
        zp_loss = self.sp_module.compute_zp_loss(obs_t, act_onehot, obs_tp1)

        return zp_loss

    def do_scheduling(self):
        self.step += 1
        step = self.step
        self.obs[step] = self.next_obs
        self.dones[step] = self.next_done

        use_pref_action = False
        if self.use_preference_algo and self.preference_estimator:
            stats = self.env.sim.get_last_interval_stats()
            self.preference_estimator.update(stats, self.env.sim)
            self.total_steps_taken += 1

            warmup_done = self.total_steps_taken >= self.preference_warmup_steps
            ready = self.preference_estimator.is_fitted
            use_pref_action = warmup_done and ready

            if use_pref_action:
                pref_actions = self.preference_estimator.get_action(self.env.sim)
                action = torch.from_numpy(pref_actions).unsqueeze(0).to(device)
                
                logprob = torch.zeros(1).to(device)
                value = torch.zeros(1).to(device)
                step_mask = torch.ones(1, self.env.sum_action).to(device)
                if not self.preference_switched:
                    self.preference_switched = True
                    print(f"[Preference Algo] Switched to estimator at step={self.total_steps_taken}, buffer={len(self.preference_estimator.replay_buffer)}")
            else:
                with torch.no_grad():
                    action, logprob, _, value, step_mask = self.agent.get_action_and_value(
                        self.next_obs.unsqueeze(0), deterministic=self.args.eval_mode)
        else:
            with torch.no_grad():
                action, logprob, _, value, step_mask = self.agent.get_action_and_value(
                    self.next_obs.unsqueeze(0), deterministic=self.args.eval_mode)
        
        self.values[step] = value.flatten()
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.action_masks[step] = step_mask

        next_obs, reward, done = self.env.step(action.cpu().numpy())
        self.rewards[step] = torch.tensor(reward).to(device).view(-1)
        self.next_obs, self.next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)

        current_time = ns.Simulator.Now().GetSeconds()
        if current_time + self.config.time_slot_duration < self.episode_duration:
            event = ns.cppyy.gbl.pythonMakeEvent(scheduling_callback, sys.argv)
            ns.Simulator.Schedule(ns.Seconds(self.config.time_slot_duration), event)

    def run(self):
        global g_sim
        g_sim = self
        args = self.args
        run_name = f"{self.config.myscheme}_{self.config.cc}_{self.config.use_self_predictive}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if args.track:
            import wandb
            wandb.init(
                project=args.project_name,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        logdir = f'./{args.project_name}/{args.out_subdir}/{run_name}'
        self.tsboard = SummaryWriter(logdir)
        self.tsboard.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

        self.global_step = 0
        self.next_obs = torch.zeros(self.env.single_observation_space).to(device)
        self.next_done = torch.zeros(1).to(device)
        if args.load_model_path:
            self.load_checkpoint(args.load_model_path)
        
        if args.eval_mode:
            self.agent.eval()
            if self.use_self_predictive:
                self.sp_module.eval()
            print("Running in Evaluation Mode (No Training)")

        file_path = f'./{args.project_name}/{args.out_subdir}/NOPC.csv'
        with open(file_path, 'a', newline='') as csvfile:
            ww = csv.writer(csvfile)
            for update in range(1, args.num_updates + 1):
                self.step = 0
                if args.anneal_lr and not args.eval_mode:
                    frac = 1.0 - (update - 1.0) / args.num_updates
                    lrnow = frac * args.learning_rate
                    self.optimizer.param_groups[0]["lr"] = lrnow
                    if self.use_self_predictive:
                        self.sp_model_optimizer.param_groups[0]["lr"] = frac * self.sp_lr

                ns.Simulator.Destroy()
                self.env.reset()
                event = ns.cppyy.gbl.pythonMakeEvent(scheduling_callback, sys.argv)
                ns.Simulator.Schedule(ns.Seconds(1.0), event)
                ns.Simulator.Stop(ns.Seconds(self.episode_duration))
                ns.Simulator.Run()

                metrics = self.env.get_metrics()
                metrics['reward'] = self.rewards.sum().item() / self.config.simulation_time

                if self.metrics is None:
                    self.metrics = metrics
                else:
                    for key, value in metrics.items():
                        self.metrics[key] += value

                if update % args.log_interval == 0:
                    averaged = {}
                    for key, value in self.metrics.items():
                        averaged[key] = value / args.log_interval
                        self.tsboard.add_scalar(f'metrics/{key}', averaged[key], update)
                    if args.track:
                        wandb.log(averaged, step=update)
                    
                    # Print metrics for visibility in eval mode
                    if args.eval_mode:
                        print(f"Eval Step {update}: {averaged}")

                    self.metrics = None
                
                if args.eval_mode or self.use_preference_algo:
                    continue

                with torch.no_grad():
                    next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
                    if args.gae:
                        advantages = torch.zeros_like(self.rewards).to(device)
                        lastgaelam = 0
                        for t in reversed(range(args.num_steps)):
                            if t == args.num_steps - 1:
                                nextnonterminal = 1.0 - self.next_done
                                nextvalues = next_value
                            else:
                                nextnonterminal = 1.0 - self.dones[t + 1]
                                nextvalues = self.values[t + 1]
                            delta = self.rewards[t] + args.gamma * \
                                nextvalues * nextnonterminal - self.values[t]
                            advantages[t] = lastgaelam = delta + args.gamma * \
                                args.gae_lambda * nextnonterminal * lastgaelam
                        returns = advantages + self.values
                    else:
                        returns = torch.zeros_like(self.rewards).to(device)
                        for t in reversed(range(args.num_steps)):
                            if t == args.num_steps - 1:
                                nextnonterminal = 1.0 - self.next_done
                                next_return = next_value
                            else:
                                nextnonterminal = 1.0 - self.dones[t + 1]
                                next_return = returns[t + 1]
                            returns[t] = self.rewards[t] + args.gamma * \
                                nextnonterminal * next_return
                        advantages = returns - self.values

                # flatten the batch
                b_obs = self.obs.reshape(-1, self.env.single_observation_space)
                b_logprobs = self.logprobs.reshape(-1)
                b_actions = self.actions.reshape(-1, self.env.single_action_space)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = self.values.reshape(-1)
                b_action_masks = self.action_masks.reshape((-1, self.action_masks.shape[-1]))

                # Optimizing the policy and value network
                b_inds = np.arange(args.batch_size)
                clipfracs = []
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, args.batch_size, args.minibatch_size):
                        end = start + args.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                            b_obs[mb_inds],
                            b_action_masks[mb_inds],
                            b_actions.long()[mb_inds].T,
                        )
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() >
                                        args.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        if args.norm_adv:
                            mb_advantages = (
                                mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * \
                            torch.clamp(ratio, 1 - args.clip_coef,
                                        1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if args.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(
                                v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * \
                                ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()

                        # ====== Combined Loss: GradNorm-Lite or Fixed ======
                        ppo_loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                        if self.use_self_predictive:
                            sp_loss = self._compute_self_predictive_loss(
                                b_obs[mb_inds], b_actions[mb_inds]
                            )

                            # GradNorm-Lite: read current weights, form weighted sum
                            gn_w = self.gradnorm.get_weights()
                            loss = gn_w[0] * ppo_loss + gn_w[1] * sp_loss
                        else:
                            loss = ppo_loss
                            sp_loss = torch.tensor(0.0, device=device)

                        # ---- Standard backward + step (no create_graph!) ----
                        self.optimizer.zero_grad()
                        if self.use_self_predictive:
                            self.sp_model_optimizer.zero_grad()

                        loss.backward()

                        nn.utils.clip_grad_norm_(
                            self.agent.parameters(), args.max_grad_norm)
                        if self.use_self_predictive:
                            nn.utils.clip_grad_norm_(
                                self.sp_module.transition_model.parameters(),
                                args.max_grad_norm)

                        self.optimizer.step()
                        if self.use_self_predictive:
                            self.sp_model_optimizer.step()
                            self.gradnorm.step(
                                losses=[ppo_loss.item(), sp_loss.item()],
                                shared_params=self.shared_params_for_gradnorm,
                            )

                    if args.target_kl is not None:
                        if approx_kl > args.target_kl:
                            break

                # ====== EMA update for target encoder ======
                if self.use_self_predictive:
                    self.sp_module.update_target_encoder()

                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - \
                    np.var(y_true - y_pred) / var_y

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                self.tsboard.add_scalar("losses/value_loss", v_loss.item(), update)
                self.tsboard.add_scalar("losses/policy_loss", pg_loss.item(), update)
                if self.use_self_predictive:
                    self.tsboard.add_scalar("losses/zp_loss", sp_loss.item(), update)
                    # # ====== GradNorm-Lite Logging ======
                    # if self.use_self_predictive:
                    #     gn_weights = self.gradnorm.get_weights_list()
                    #     self.tsboard.add_scalar("gradnorm/weight_ppo", gn_weights[0], update)
                    #     self.tsboard.add_scalar("gradnorm/weight_zp", gn_weights[1], update)
                
                if args.save_model and update % args.save_interval == 0:
                    model_path = os.path.join(logdir, f"model_{update}.pth")
                    self.save_checkpoint(model_path)
        
        if args.save_model:
            model_path = os.path.join(logdir, "model_1000.pth")
            self.save_checkpoint(model_path)

        self.tsboard.close()
        ns.Simulator.Destroy()


if __name__ == "__main__":
    args = parse_args()
    config = Config()
    env = EnvForDRL(config)
    main = Main(args, config, env)
    main.run()