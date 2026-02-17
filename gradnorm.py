import torch
import torch.nn as nn

class GradNormBalancer:
    """
    GradNorm-Lite: Automatic Multi-task Loss Balancing
    
    Dynamically balances multiple task losses by adjusting task weights
    to equalize gradient magnitudes across tasks.
    """
    def __init__(self, num_tasks, alpha=1.5, lr=0.025, device='cpu'):
        """
        Args:
            num_tasks: Number of tasks to balance
            alpha: Restoring force strength (0-3, typical 1.0-1.5)
                  Higher alpha gives more weight to slower-training tasks
            lr: Learning rate for weight updates
            device: Device to place tensors on
        """
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.lr = lr
        self.device = device
        
        # Initialize task weights (all equal at start)
        self.weights = torch.ones(num_tasks, device=device, requires_grad=False)
        
        # Track initial losses and gradient norms
        self._initial_losses = None
        self._loss_ema = None
        self._grad_norm_ema = None
        self.ema_momentum = 0.9
    
    def get_weights(self):
        """Return current task weights as a tensor"""
        return self.weights
    
    def get_weights_list(self):
        """Return current task weights as a list"""
        return self.weights.tolist()
    
    def step(self, losses, shared_params):
        """
        Update task weights based on gradient norms
        
        Args:
            losses: List or tensor of loss values for each task [num_tasks]
            shared_params: List of shared parameters (e.g., encoder parameters)
        """
        losses = torch.tensor(losses, device=self.device) if not isinstance(losses, torch.Tensor) else losses
        
        # Initialize tracking on first step
        if self._initial_losses is None:
            self._initial_losses = losses.detach().clone()
            self._loss_ema = losses.detach().clone()
            self._grad_norm_ema = torch.zeros(self.num_tasks, device=self.device)
            return
        
        # Update loss EMA
        self._loss_ema = self.ema_momentum * self._loss_ema + (1 - self.ema_momentum) * losses.detach()
        
        # Compute gradient norms for each task
        grad_norms = []
        for i in range(self.num_tasks):
            # Compute gradients for task i
            task_grads = torch.autograd.grad(
                self.weights[i] * losses[i],
                shared_params,
                retain_graph=True,
                create_graph=False
            )
            
            # Compute L2 norm of gradients
            grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in task_grads if g is not None))
            grad_norms.append(grad_norm.item())
        
        grad_norms = torch.tensor(grad_norms, device=self.device)
        
        # Update gradient norm EMA
        self._grad_norm_ema = self.ema_momentum * self._grad_norm_ema + (1 - self.ema_momentum) * grad_norms
        
        # Compute inverse training rates (higher = slower training)
        with torch.no_grad():
            loss_ratios = self._loss_ema / (self._initial_losses + 1e-8)
            inverse_train_rates = loss_ratios ** self.alpha
            
            # Compute mean gradient norm
            mean_grad_norm = self._grad_norm_ema.mean()
            
            # Compute target gradient norms
            target_grad_norms = mean_grad_norm * inverse_train_rates
            
            # Update weights to move gradients toward targets
            grad_norm_ratios = target_grad_norms / (self._grad_norm_ema + 1e-8)
            self.weights = self.weights * (1 + self.lr * (grad_norm_ratios - 1))
            
            # Renormalize weights to sum to num_tasks (keeps magnitude stable)
            self.weights = self.weights * (self.num_tasks / self.weights.sum())
            
            # Clamp weights to reasonable range
            self.weights = torch.clamp(self.weights, 0.1, 10.0)
