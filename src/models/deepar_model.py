"""DeepAR PyTorch neural network architecture.

Implements an autoregressive LSTM network with static categorical embeddings,
dynamic calendar covariates, real-time mean scaling, and a Negative Binomial
distribution head for retail count demand.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Normal

from src.models.deepar_config import DeepARConfig


def get_device(preference: str = "auto") -> torch.device:
    """Select the best available compute device according to preference."""
    pref = preference.lower().strip()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


class NegativeBinomialHead(nn.Module):
    """Negative Binomial distribution parameter head.

    Parameterizes count distribution by mean mu > 0 and dispersion alpha > 0,
    such that Var(Y) = mu + alpha * mu^2.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_alpha = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distribution parameters (mu, alpha).

        Args:
            h: Recurrent hidden states of shape (batch, seq_len, hidden_dim).
            scale: Scale factor nu of shape (batch,) or (batch, 1) or (batch, seq_len).

        Returns:
            mu: Mean demand of shape (batch, seq_len).
            alpha: Dispersion parameter of shape (batch, seq_len).
        """
        # Ensure scale can broadcast with (batch, seq_len)
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)
        elif scale.dim() == 2 and scale.size(1) == 1:
            pass

        # Unscaled positive mean via softplus
        raw_mu = F.softplus(self.fc_mu(h)).squeeze(-1)
        mu = raw_mu * scale + 1e-5

        # Dispersion parameter alpha > 0 (overdispersion factor)
        alpha = F.softplus(self.fc_alpha(h)).squeeze(-1) + 1e-4

        return mu, alpha

    def distribution(self, mu: torch.Tensor, alpha: torch.Tensor) -> NegativeBinomial:
        """Create PyTorch NegativeBinomial distribution instance.

        PyTorch NegativeBinomial(total_count=r, logits=logits)
        r = 1 / alpha
        logits = log(alpha * mu)
        """
        r = 1.0 / alpha
        # logits = log(p / (1-p)) = log(alpha * mu)
        logits = torch.log(alpha * mu + 1e-8)
        return NegativeBinomial(total_count=r, logits=logits)

    def sample(self, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Sample from Negative Binomial distribution, handling MPS fallback cleanly."""
        device = mu.device
        if device.type == "mps":
            # MPS lacks native _standard_gamma kernel; execute sample on CPU
            mu_cpu = mu.cpu()
            alpha_cpu = alpha.cpu()
            dist = self.distribution(mu_cpu, alpha_cpu)
            samples = dist.sample()
            return samples.to(device)
        else:
            dist = self.distribution(mu, alpha)
            return dist.sample()


class GaussianHead(nn.Module):
    """Gaussian distribution parameter head with mean scaling."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)
        mu = self.fc_mu(h).squeeze(-1) * scale
        sigma = F.softplus(self.fc_sigma(h)).squeeze(-1) * scale + 1e-3
        return mu, sigma

    def distribution(self, mu: torch.Tensor, sigma: torch.Tensor) -> Normal:
        return Normal(loc=mu, scale=sigma)

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(mu, sigma)
        return dist.sample()


class DeepARNetwork(nn.Module):
    """Complete DeepAR autoregressive probabilistic forecasting neural network."""

    def __init__(
        self,
        config: DeepARConfig,
        vocab_sizes: Dict[str, int],
        covariate_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_sizes = vocab_sizes
        self.covariate_dim = covariate_dim

        # 1. Static categorical embeddings
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        if config.use_static_features and vocab_sizes:
            for cat_name, vocab_size in vocab_sizes.items():
                embed_dim = config.embedding_dim_map.get(cat_name, 4)
                self.embeddings[cat_name] = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embed_dim,
                    padding_idx=0,
                )
                total_embed_dim += embed_dim
        self.total_embed_dim = total_embed_dim

        # 2. Input dimension: 1 (scaled target) + embeddings + dynamic covariates
        effective_cov_dim = covariate_dim if config.use_calendar_features else 0
        self.input_dim = 1 + total_embed_dim + effective_cov_dim

        # 3. Recurrent core: Multi-layer LSTM
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        # 4. Output distribution head
        if config.distribution == "negative_binomial":
            self.head = NegativeBinomialHead(config.hidden_dim)
        elif config.distribution == "gaussian":
            self.head = GaussianHead(config.hidden_dim)
        else:
            raise ValueError(f"Unknown distribution: {config.distribution}")

    def _embed_static(self, static_cat: torch.Tensor) -> Optional[torch.Tensor]:
        """Embed static categorical features."""
        if not self.config.use_static_features or not self.embeddings:
            return None
        embed_list = []
        for col_idx, (cat_name, layer) in enumerate(self.embeddings.items()):
            col_tensor = static_cat[:, col_idx]
            embed_list.append(layer(col_tensor))
        return torch.cat(embed_list, dim=-1)

    def _build_step_input(
        self,
        y_scaled: torch.Tensor,
        static_embed: Optional[torch.Tensor],
        covariates: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Concatenate scaled previous target, static embeddings, and covariates."""
        parts = [y_scaled.unsqueeze(-1)]
        if static_embed is not None:
            parts.append(static_embed)
        if covariates is not None and self.config.use_calendar_features and covariates.size(-1) > 0:
            parts.append(covariates)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        target: torch.Tensor,
        covariates: torch.Tensor,
        static_cat: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass over full sequence with teacher forcing during training.

        Args:
            target: Ground truth sequence of shape (batch, seq_len).
            covariates: Dynamic covariates of shape (batch, seq_len, cov_dim).
            static_cat: Static categorical codes of shape (batch, num_cats).
            scale: Scale factor nu of shape (batch,).

        Returns:
            loss: Negative log-likelihood scalar loss.
            dist_params: Tuple of (param1, param2) distributions across time.
        """
        batch_size, seq_len = target.shape

        # Construct lagged target inputs: y_{t-1}
        # For t=0, lag is 0.0
        y_prev = torch.cat([torch.zeros(batch_size, 1, device=target.device), target[:, :-1]], dim=1)
        y_scaled = torch.log1p(y_prev / scale.unsqueeze(1))

        # Embed static features and expand along sequence length
        static_embed = self._embed_static(static_cat)
        static_seq = static_embed.unsqueeze(1).expand(-1, seq_len, -1) if static_embed is not None else None

        # Build LSTM input tensor of shape (batch, seq_len, input_dim)
        x = self._build_step_input(y_scaled, static_seq, covariates if self.config.use_calendar_features else None)

        # Run LSTM
        h, _ = self.lstm(x)

        # Predict distribution parameters at all time steps
        param1, param2 = self.head(h, scale)
        dist = self.head.distribution(param1, param2)

        # Compute negative log-likelihood loss
        # Target values must be nonnegative
        target_clamped = torch.clamp(target, min=0.0)
        log_prob = dist.log_prob(target_clamped)

        # Loss is average negative log-likelihood per element
        loss = -log_prob.mean()

        return loss, (param1, param2)

    def forecast(
        self,
        context_targets: torch.Tensor,
        context_covs: torch.Tensor,
        future_covs: torch.Tensor,
        static_cats: torch.Tensor,
        scales: torch.Tensor,
        num_samples: int = 200,
    ) -> torch.Tensor:
        """Autoregressive Monte Carlo probabilistic forecasting.

        Args:
            context_targets: Historical target observations of shape (batch, context_len).
            context_covs: Historical covariates of shape (batch, context_len, cov_dim).
            future_covs: Known future covariates of shape (batch, horizon, cov_dim).
            static_cats: Static categorical codes of shape (batch, num_cats).
            scales: Per-series scale factor nu of shape (batch,).
            num_samples: Number of sample trajectories S to simulate.

        Returns:
            samples: Simulated demand trajectories of shape (batch, num_samples, horizon).
        """
        self.eval()
        batch_size, context_len = context_targets.shape
        horizon = future_covs.shape[1]
        device = context_targets.device

        with torch.no_grad():
            # 1. Unroll context window with teacher forcing to get the final recurrent state
            y_prev_ctx = torch.cat(
                [torch.zeros(batch_size, 1, device=device), context_targets[:, :-1]], dim=1
            )
            y_scaled_ctx = torch.log1p(y_prev_ctx / scales.unsqueeze(1))

            static_embed = self._embed_static(static_cats)
            static_seq_ctx = (
                static_embed.unsqueeze(1).expand(-1, context_len, -1)
                if static_embed is not None
                else None
            )

            x_ctx = self._build_step_input(
                y_scaled_ctx,
                static_seq_ctx,
                context_covs if self.config.use_calendar_features else None,
            )

            _, (h_n, c_n) = self.lstm(x_ctx)

            # 2. Expand recurrent state and inputs for parallel Monte Carlo sampling
            # Total parallel instances: B * S
            B_S = batch_size * num_samples

            # Repeat interleave ensures sample trajectories for the same series are grouped
            h_roll = h_n.repeat_interleave(num_samples, dim=1)  # (num_layers, B*S, hidden_dim)
            c_roll = c_n.repeat_interleave(num_samples, dim=1)
            scales_roll = scales.repeat_interleave(num_samples) # (B*S,)

            static_roll = (
                static_embed.repeat_interleave(num_samples, dim=0)
                if static_embed is not None
                else None
            )

            # Initial autoregressive input at horizon step 1 is the last context target
            y_prev = context_targets[:, -1].repeat_interleave(num_samples)  # (B*S,)

            samples_history = []

            # 3. Autoregressive sequential generation over forecast horizon
            for step in range(horizon):
                y_scaled_step = torch.log1p(y_prev / scales_roll).unsqueeze(1)  # (B*S, 1)

                if self.config.use_calendar_features and future_covs.size(-1) > 0:
                    cov_step = future_covs[:, step:step + 1, :].repeat_interleave(num_samples, dim=0)
                else:
                    cov_step = None

                static_step = static_roll.unsqueeze(1) if static_roll is not None else None
                x_step = self._build_step_input(y_scaled_step, static_step, cov_step)

                # Step LSTM forward by 1 time step
                out_step, (h_roll, c_roll) = self.lstm(x_step, (h_roll, c_roll))

                # Compute distribution parameters
                param1, param2 = self.head(out_step, scales_roll.unsqueeze(1))

                # Draw sample from predictive distribution
                sample_step = self.head.sample(param1.squeeze(1), param2.squeeze(1))

                # Clamping ensures strictly non-negative demand values
                sample_step = torch.clamp(sample_step, min=0.0)

                # Feed sample into the next autoregressive step
                y_prev = sample_step
                samples_history.append(sample_step)

            # Stack along horizon dimension: (horizon, B*S) -> (B*S, horizon)
            all_samples = torch.stack(samples_history, dim=1)

            # Reshape to (batch_size, num_samples, horizon)
            all_samples = all_samples.view(batch_size, num_samples, horizon)

            return all_samples

