"""
Unit tests for Component B: LSTM Model (lstm.py)
"""
import pytest
import torch

from src.model.lstm import (
    PushupLSTM,
    PushupLSTMLoss,
    create_model,
    load_checkpoint,
    save_checkpoint,
)


class TestPushupLSTM:
    """Tests for PushupLSTM model."""

    def test_model_creation(self):
        """Test model can be created with default params (causal/unidirectional)."""
        model = create_model()
        assert isinstance(model, PushupLSTM)
        assert model.input_dim == 19
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.num_classes == 3
        # Default is unidirectional so the model is causal (real-time capable).
        assert model.bidirectional is False

    def test_model_custom_params(self):
        """Test model creation with custom params."""
        model = create_model(
            input_dim=10,
            hidden_size=32,
            num_layers=1,
            num_classes=3,
            dropout=0.1,
        )
        assert model.input_dim == 10
        assert model.hidden_size == 32
        assert model.num_layers == 1
        # Check classifier dropout (LSTM dropout is 0.0 when num_layers=1)
        assert model.classifier[2].p == 0.1

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = create_model()
        batch_size = 4
        seq_len = 50
        input_dim = 19

        x = torch.randn(batch_size, seq_len, input_dim)
        logits = model(x)

        assert logits.shape == (batch_size, seq_len, 3)

    def test_forward_with_lengths(self):
        """Test forward pass with packed sequences."""
        model = create_model()
        batch_size = 3
        max_len = 40
        input_dim = 19

        # Variable length sequences
        x = torch.randn(batch_size, max_len, input_dim)
        lengths = torch.tensor([20, 35, 40])

        logits = model(x, lengths)
        assert logits.shape == (batch_size, max_len, 3)

    def test_forward_variable_lengths(self):
        """Test that different lengths produce correct outputs."""
        model = create_model()
        model.eval()

        x = torch.randn(2, 30, 19)
        lengths = torch.tensor([15, 30])

        with torch.no_grad():
            logits = model(x, lengths)

        # First sequence should have meaningful outputs only up to length 15
        # (though pad_padded_sequence fills rest with zeros)
        assert logits.shape == (2, 30, 3)

    def test_get_hidden_states(self):
        """Test hidden state extraction."""
        model = create_model()
        x = torch.randn(2, 20, 19)
        lengths = torch.tensor([15, 20])

        hn, cn = model.get_hidden_states(x, lengths)

        # Unidirectional LSTM with 2 layers: num_directions=1, num_layers=2
        assert hn.shape == (2, 2, 64)  # (num_layers * 1, batch, hidden)
        assert cn.shape == (2, 2, 64)

    def test_bidirectional_hidden_states(self):
        """Test hidden state shape for a bidirectional model."""
        model = create_model(bidirectional=True)
        x = torch.randn(2, 20, 19)
        lengths = torch.tensor([15, 20])

        hn, cn = model.get_hidden_states(x, lengths)

        # BiLSTM with 2 layers: num_directions=2, num_layers=2
        assert hn.shape == (4, 2, 64)
        assert cn.shape == (4, 2, 64)

    def test_step_streaming(self):
        """Test the causal streaming step() matches forward on the same input."""
        torch.manual_seed(0)
        model = create_model(bidirectional=False)
        model.eval()
        T = 25
        x = torch.randn(1, T, 19)

        # Batch forward
        with torch.no_grad():
            batch_logits = model(x, torch.tensor([T]))

        # Stepped forward
        hidden = None
        step_logits = []
        with torch.no_grad():
            for t in range(T):
                logit, hidden = model.step(x[:, t:t + 1], hidden)
                step_logits.append(logit)
        step_logits = torch.cat(step_logits, dim=1)

        assert step_logits.shape == (1, T, 3)
        assert torch.allclose(batch_logits, step_logits, atol=1e-4)

    def test_step_rejects_bidirectional(self):
        """Streaming step() must refuse a bidirectional (non-causal) model."""
        model = create_model(bidirectional=True)
        x = torch.randn(1, 1, 19)
        with pytest.raises(ValueError):
            model.step(x)


class TestPushupLSTMLoss:
    """Tests for loss function."""

    def test_loss_basic(self):
        """Test basic loss computation."""
        criterion = PushupLSTMLoss()
        logits = torch.randn(2, 10, 3)
        targets = torch.randint(0, 3, (2, 10))

        loss = criterion(logits, targets)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_loss_with_ignore_index(self):
        """Test loss ignores padded positions."""
        criterion = PushupLSTMLoss(ignore_index=-1)
        logits = torch.randn(2, 10, 3)
        targets = torch.tensor([
            [0, 1, 2, 1, 0, -1, -1, -1, -1, -1],
            [1, 2, 1, 0, 1, 2, 0, -1, -1, -1],
        ])

        loss = criterion(logits, targets)
        assert loss.item() >= 0

    def test_loss_with_lengths(self):
        """Test loss with explicit lengths."""
        criterion = PushupLSTMLoss(ignore_index=-1)
        logits = torch.randn(2, 10, 3)
        targets = torch.randint(0, 3, (2, 10))
        lengths = torch.tensor([5, 8])

        loss = criterion(logits, targets, lengths)
        assert loss.item() >= 0

    def test_loss_label_smoothing(self):
        """Test loss with label smoothing."""
        criterion = PushupLSTMLoss(label_smoothing=0.1)
        logits = torch.randn(2, 10, 3)
        targets = torch.randint(0, 3, (2, 10))

        loss = criterion(logits, targets)
        assert loss.item() >= 0


class TestCheckpointSaveLoad:
    """Tests for checkpoint saving and loading."""

    def test_save_load_checkpoint(self, tmp_path):
        """Test saving and loading a checkpoint."""
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        config = {"input_dim": 19, "hidden_size": 64}
        metrics = {"val_loss": 0.5, "macro_f1": 0.85}

        save_checkpoint(
            str(checkpoint_path),
            model, optimizer, epoch=10, metrics=metrics, config=config
        )

        # Load into new model
        new_model = create_model()

        checkpoint = load_checkpoint(str(checkpoint_path), new_model)

        assert checkpoint['epoch'] == 10
        assert checkpoint['metrics']['val_loss'] == 0.5
        assert checkpoint['config']['input_dim'] == 19

        # Check weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint raises error."""
        model = create_model()
        with pytest.raises(Exception):
            load_checkpoint("/nonexistent/path.pt", model)


class TestModelDeterministic:
    """Test model produces deterministic outputs with fixed seed."""

    def test_deterministic_forward(self):
        """Same input should give same output with fixed seed."""
        torch.manual_seed(42)
        model1 = create_model()
        model1.eval()

        torch.manual_seed(42)
        model2 = create_model()
        model2.eval()

        x = torch.randn(2, 20, 19)

        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
