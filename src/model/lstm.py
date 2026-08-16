"""
Component B: LSTM Sequence Model for Push-up Phase Classification

BiLSTM with per-timestep 3-class classification head (idle, concentric, eccentric).
Handles variable-length sequences with packed sequences.
"""
import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


class PushupLSTM(nn.Module):
    """
    LSTM for exercise phase classification.

    Input: (batch, seq_len, feature_dim) with variable lengths
    Output: (batch, seq_len, 3) logits for [idle, concentric, eccentric]

    By default the LSTM is unidirectional so that it is *causal*: every output
    depends only on past frames, which is what the real-time streaming path
    requires (``step`` feeds one frame at a time). A bidirectional variant is
    still available for offline use.
    """

    def __init__(
        self,
        input_dim: int = 19,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Classification head
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM and classifier weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (common practice)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            lengths: Optional tensor of actual sequence lengths (batch,)

        Returns:
            Logits of shape (batch, seq_len, num_classes)
        """
        seq_len = x.shape[1]

        if lengths is not None:
            # Pack padded sequence for efficient computation
            # Ensure lengths are on CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
        else:
            output, _ = self.lstm(x)

        # Apply classifier to each timestep
        logits = self.classifier(output)  # (batch, seq_len, num_classes)

        return logits

    def get_hidden_states(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get final hidden states (for potential downstream tasks).
        Returns: (hn, cn) where hn shape is (num_layers * num_directions, batch, hidden_size)
        """
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            _, (hn, cn) = self.lstm(packed)
        else:
            _, (hn, cn) = self.lstm(x)
        return hn, cn

    def step(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Causal single-timestep forward pass for real-time streaming.

        Args:
            x: (batch, 1, input_dim) — exactly one timestep.
            hidden: optional (h, c) tuple from the previous ``step`` call.

        Returns:
            (logits, hidden) where logits has shape (batch, 1, num_classes)
            and hidden is the (h, c) tuple to feed back into the next step.

        Only valid for a unidirectional LSTM: a bidirectional model reads the
        future, so it cannot be streamed. The hidden state shape for a
        unidirectional, num_layers-deep LSTM is (num_layers, batch, hidden).
        """
        if self.bidirectional:
            raise ValueError(
                "step() is only available for a unidirectional LSTM; "
                "streaming is causal and cannot use future frames."
            )
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(x, hidden)
        logits = self.classifier(out)  # (batch, 1, num_classes)
        return logits, hidden


class PushupLSTMLoss(nn.Module):
    """
    Loss function for push-up LSTM with masking for padded positions.
    Uses CrossEntropyLoss with ignore_index for padded timesteps.
    """

    def __init__(self, ignore_index: int = -1, label_smoothing: float = 0.1):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute loss with masking.

        Args:
            logits: (batch, seq_len, num_classes)
            targets: (batch, seq_len) with class indices, padded positions = ignore_index
            lengths: Optional actual lengths for additional masking

        Returns:
            Scalar loss
        """
        batch_size, seq_len, num_classes = logits.shape

        # Reshape for CrossEntropyLoss: (batch * seq_len, num_classes) and (batch * seq_len,)
        logits_flat = logits.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)

        # If lengths provided, create mask for padded positions
        if lengths is not None:
            mask = torch.arange(seq_len, device=lengths.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
            # Set masked positions to ignore_index
            targets_flat = targets_flat.masked_fill(~mask.reshape(-1), self.ignore_index)

        loss = self.ce_loss(logits_flat, targets_flat)
        return loss


def masked_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Per-timestep classification accuracy over non-padded positions only.

    Args:
        logits: (batch, seq_len, num_classes)
        targets: (batch, seq_len) with class indices; padded positions should
                 already be -1 (the ignore index) or are masked out via lengths.
        lengths: Optional (batch,) tensor of real sequence lengths.

    Returns:
        Scalar accuracy in [0, 1].
    """
    batch_size, seq_len, _ = logits.shape
    preds = torch.argmax(logits, dim=2)

    if lengths is not None:
        mask = torch.arange(seq_len, device=lengths.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        targets = targets.masked_fill(~mask, -1)

    valid = targets != -1
    if valid.sum().item() == 0:
        return torch.tensor(0.0, device=logits.device)
    return (preds[valid] == targets[valid]).float().mean()


def create_model(
    input_dim: int = 19,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_classes: int = 3,
    dropout: float = 0.2,
    device: str = "cpu",
    bidirectional: bool = False,
) -> PushupLSTM:
    """Factory function to create model with given config."""
    model = PushupLSTM(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    return model.to(device)


def load_checkpoint(
    path: str,
    model: PushupLSTM,
    device: str = "cpu",
) -> dict:
    """
    Load model checkpoint.

    Returns:
        Dict with keys: model_state_dict, optimizer_state_dict, epoch, metrics, config
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def save_checkpoint(
    path: str,
    model: PushupLSTM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
):
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'config': config,
    }, path)
