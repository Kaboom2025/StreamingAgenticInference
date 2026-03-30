"""Tests for LLM backend implementations (llama-cpp, HuggingFace, MLX)."""

import ctypes
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("llama_cpp")

from streamagent.engine.backends.factory import create_backend
from streamagent.engine.backends.hf_backend import HFBackend
from streamagent.engine.backends.llama_backend import LlamaBackend
from streamagent.engine.backends.mlx_backend import MLXBackend
from streamagent.engine.interfaces import BackendProtocol


def _ctypes_float_array(*values: float) -> "ctypes.Array[ctypes.c_float]":
    """Create a ctypes float array for mocking llama_get_logits return values."""
    arr_type = ctypes.c_float * len(values)
    return arr_type(*values)


# ---------------------------------------------------------------------------
# LlamaBackend Tests
# ---------------------------------------------------------------------------


class TestLlamaBackendNotLoaded:
    """Test LlamaBackend raises errors before model is loaded."""

    def test_tokenize_raises_without_load(self):
        """Calling tokenize before load_model raises RuntimeError."""
        backend = LlamaBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.tokenize("hello")

    def test_detokenize_raises_without_load(self):
        """Calling detokenize before load_model raises RuntimeError."""
        backend = LlamaBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.detokenize([1, 2, 3])

    def test_prefill_raises_without_load(self):
        """Calling prefill before load_model raises RuntimeError."""
        backend = LlamaBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.prefill([1, 2, 3])

    def test_forward_one_raises_without_load(self):
        """Calling forward_one before load_model raises RuntimeError."""
        backend = LlamaBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.forward_one(token_id=42, cache_position=0)

    def test_context_length_raises_without_load(self):
        """Accessing context_length before load_model raises RuntimeError."""
        backend = LlamaBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = backend.context_length


class TestLlamaBackendLoad:
    """Test LlamaBackend model loading and configuration."""

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_load_model_calls_llama_constructor(self, mock_llama_class):
        """load_model calls Llama constructor with correct parameters."""
        mock_instance = MagicMock()
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model(
            "/path/to/model.gguf", n_ctx=2048, n_gpu_layers=10
        )

        mock_llama_class.assert_called_once_with(
            model_path="/path/to/model.gguf",
            n_ctx=2048,
            n_gpu_layers=10,
            logits_all=False,
            verbose=False,
        )

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_load_model_default_n_ctx(self, mock_llama_class):
        """load_model uses default n_ctx=4096 if not specified."""
        mock_instance = MagicMock()
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")

        # Check that n_ctx was set to 4096
        call_args = mock_llama_class.call_args
        assert call_args[1]["n_ctx"] == 4096

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_load_model_default_n_gpu_layers(self, mock_llama_class):
        """load_model uses default n_gpu_layers=-1 if not specified."""
        mock_instance = MagicMock()
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")

        call_args = mock_llama_class.call_args
        assert call_args[1]["n_gpu_layers"] == -1

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_load_model_sets_context_length(self, mock_llama_class):
        """load_model stores context_length from n_ctx parameter."""
        mock_instance = MagicMock()
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf", n_ctx=8192)

        assert backend.context_length == 8192

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_load_model_context_length_default(self, mock_llama_class):
        """context_length returns 4096 when n_ctx not specified."""
        mock_instance = MagicMock()
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")

        assert backend.context_length == 4096


class TestLlamaBackendTokenization:
    """Test LlamaBackend tokenization methods."""

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_tokenize_calls_model_tokenize(self, mock_llama_class):
        """tokenize calls model.tokenize with encoded text."""
        mock_instance = MagicMock()
        mock_instance.tokenize.return_value = [1, 2, 3]
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")
        result = backend.tokenize("hello")

        mock_instance.tokenize.assert_called_once_with(b"hello", special=True)
        assert result == [1, 2, 3]

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_detokenize_calls_model_detokenize(self, mock_llama_class):
        """detokenize calls model.detokenize with token ids."""
        mock_instance = MagicMock()
        mock_instance.detokenize.return_value = b"hello"
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")
        result = backend.detokenize([1, 2, 3])

        mock_instance.detokenize.assert_called_once_with([1, 2, 3])
        assert result == "hello"

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_detokenize_handles_decode_errors(self, mock_llama_class):
        """detokenize uses replace error handler for invalid UTF-8."""
        mock_instance = MagicMock()
        # Simulate invalid UTF-8 bytes
        mock_instance.detokenize.return_value = b"\xff\xfe"
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")
        result = backend.detokenize([99, 100])

        # Should not raise, should use replacement character
        assert isinstance(result, str)


class TestLlamaBackendPrefill:
    """Test LlamaBackend prefill operation."""

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_prefill_calls_eval(self, mock_llama_class):
        """prefill calls model.eval with input_ids."""
        mock_instance = MagicMock()
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")
        backend.prefill([1, 2, 3, 4])

        mock_instance.eval.assert_called_once_with([1, 2, 3, 4])

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_prefill_returns_input_length(self, mock_llama_class):
        """prefill returns the length of input_ids (cache length after prefill)."""
        mock_instance = MagicMock()
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")
        result = backend.prefill([1, 2, 3, 4, 5])

        assert result == 5

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_prefill_empty_input(self, mock_llama_class):
        """prefill with empty input returns 0."""
        mock_instance = MagicMock()
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")
        result = backend.prefill([])

        assert result == 0


class TestLlamaBackendForwardOne:
    """Test LlamaBackend single token forward pass."""

    @patch("streamagent.engine.backends.llama_backend.llama_get_logits")
    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_forward_one_calls_eval(self, mock_llama_class, mock_llama_get_logits):
        """forward_one calls model.eval with single token (second call after first is skipped)."""
        mock_instance = MagicMock()
        mock_instance.n_vocab.return_value = 3
        mock_llama_class.return_value = mock_instance

        logits = [0.5, 1.5, 2.0]
        mock_llama_get_logits.return_value = _ctypes_float_array(*logits)

        backend = LlamaBackend(temperature=0.0)  # Use greedy sampling
        backend.load_model("/path/to/model.gguf")
        backend.prefill([1, 2, 3])

        # First forward_one: skips eval (uses prefill logits)
        backend.forward_one(token_id=5, cache_position=3)
        assert len(mock_instance.eval.call_args_list) == 1  # only prefill call

        # Second forward_one: calls eval
        backend.forward_one(token_id=2, cache_position=4)
        calls = mock_instance.eval.call_args_list
        assert len(calls) == 2
        assert calls[1][0][0] == [2]

    @patch("streamagent.engine.backends.llama_backend.llama_get_logits")
    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_forward_one_returns_tuple(self, mock_llama_class, mock_llama_get_logits):
        """forward_one returns (next_token_id, log_prob) as tuple."""
        mock_instance = MagicMock()
        mock_instance.n_vocab.return_value = 3
        mock_llama_class.return_value = mock_instance

        # logits: [1.0, 2.0, 3.0] -> argmax=2, log_prob computed
        logits = [1.0, 2.0, 3.0]
        mock_llama_get_logits.return_value = _ctypes_float_array(*logits)

        backend = LlamaBackend(temperature=0.0)  # Use greedy sampling
        backend.load_model("/path/to/model.gguf")
        backend.prefill([1, 2, 3])
        next_id, log_prob = backend.forward_one(token_id=5, cache_position=3)

        assert isinstance(next_id, int)
        assert isinstance(log_prob, float)

    @patch("streamagent.engine.backends.llama_backend.llama_get_logits")
    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_forward_one_argmax_selection(self, mock_llama_class, mock_llama_get_logits):
        """forward_one returns token with highest logit."""
        mock_instance = MagicMock()
        mock_instance.n_vocab.return_value = 4
        mock_llama_class.return_value = mock_instance

        # logits: [0.5, 1.5, 2.0, 1.0] -> argmax=2
        logits = [0.5, 1.5, 2.0, 1.0]
        mock_llama_get_logits.return_value = _ctypes_float_array(*logits)

        backend = LlamaBackend(temperature=0.0)  # Use greedy sampling
        backend.load_model("/path/to/model.gguf")
        backend.prefill([1, 2, 3])
        next_id, _ = backend.forward_one(token_id=5, cache_position=3)

        assert next_id == 2

    @patch("streamagent.engine.backends.llama_backend.llama_get_logits")
    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_forward_one_log_prob_is_reasonable(self, mock_llama_class, mock_llama_get_logits):
        """forward_one log_prob is negative (log of probability < 1)."""
        mock_instance = MagicMock()
        mock_instance.n_vocab.return_value = 4
        mock_llama_class.return_value = mock_instance

        logits = [1.0, 2.0, 3.0, 1.0]
        mock_llama_get_logits.return_value = _ctypes_float_array(*logits)

        backend = LlamaBackend(temperature=0.0)  # Use greedy sampling
        backend.load_model("/path/to/model.gguf")
        backend.prefill([1, 2, 3])
        _, log_prob = backend.forward_one(token_id=5, cache_position=3)

        assert log_prob < 0, "log_prob should be negative (log of prob < 1)"


# ---------------------------------------------------------------------------
# HFBackend Tests
# ---------------------------------------------------------------------------


class TestHFBackendNotLoaded:
    """Test HFBackend raises errors before model is loaded."""

    def test_tokenize_raises_without_load(self):
        """Calling tokenize before load_model raises RuntimeError."""
        backend = HFBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.tokenize("hello")

    def test_detokenize_raises_without_load(self):
        """Calling detokenize before load_model raises RuntimeError."""
        backend = HFBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.detokenize([1, 2, 3])

    def test_prefill_raises_without_load(self):
        """Calling prefill before load_model raises RuntimeError."""
        backend = HFBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.prefill([1, 2, 3])

    def test_forward_one_raises_without_load(self):
        """Calling forward_one before load_model raises RuntimeError."""
        backend = HFBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.forward_one(token_id=42, cache_position=0)

    def test_context_length_raises_without_load(self):
        """Accessing context_length before load_model raises RuntimeError."""
        backend = HFBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = backend.context_length


class TestHFBackendLoad:
    """Test HFBackend model loading."""

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    def test_load_model_loads_tokenizer_and_model(
        self, mock_model_class, mock_tokenizer_class
    ):
        """load_model loads both tokenizer and model."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HFBackend()
        backend.load_model("gpt2")

        mock_tokenizer_class.from_pretrained.assert_called_once_with("gpt2")
        mock_model_class.from_pretrained.assert_called_once()

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    def test_load_model_uses_device_map_auto(
        self, mock_model_class, mock_tokenizer_class
    ):
        """load_model uses device_map='auto' for multi-GPU."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HFBackend()
        backend.load_model("gpt2")

        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        assert call_kwargs.get("device_map") == "auto"

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    def test_load_model_context_from_config(
        self, mock_model_class, mock_tokenizer_class
    ):
        """load_model reads context_length from model.config.max_position_embeddings."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 2048
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HFBackend()
        backend.load_model("gpt2")

        assert backend.context_length == 2048

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    def test_load_model_context_from_kwargs(
        self, mock_model_class, mock_tokenizer_class
    ):
        """load_model uses n_ctx from kwargs if provided."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 2048
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HFBackend()
        backend.load_model("gpt2", n_ctx=8192)

        assert backend.context_length == 8192


class TestHFBackendTokenization:
    """Test HFBackend tokenization."""

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    def test_tokenize_calls_encoder(self, mock_model_class, mock_tokenizer_class):
        """tokenize calls tokenizer.encode."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HFBackend()
        backend.load_model("gpt2")
        result = backend.tokenize("hello")

        mock_tokenizer.encode.assert_called_once_with(
            "hello", add_special_tokens=False
        )
        assert result == [1, 2, 3]

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    def test_detokenize_calls_decoder(
        self, mock_model_class, mock_tokenizer_class
    ):
        """detokenize calls tokenizer.decode."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "hello"
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HFBackend()
        backend.load_model("gpt2")
        result = backend.detokenize([1, 2, 3])

        mock_tokenizer.decode.assert_called_once_with(
            [1, 2, 3], skip_special_tokens=False
        )
        assert result == "hello"


class TestHFBackendPrefill:
    """Test HFBackend prefill."""

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    @patch("streamagent.engine.backends.hf_backend.torch")
    def test_prefill_stores_past_key_values(
        self, mock_torch, mock_model_class, mock_tokenizer_class
    ):
        """prefill stores past_key_values from model output."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096

        # Mock output
        mock_output = MagicMock()
        mock_past_kv = MagicMock()
        mock_output.past_key_values = mock_past_kv
        mock_model.return_value = mock_output

        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock torch tensor
        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor

        backend = HFBackend()
        backend.load_model("gpt2")
        backend.prefill([1, 2, 3])

        # Verify past_key_values is stored
        assert backend._past_key_values is not None

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    @patch("streamagent.engine.backends.hf_backend.torch")
    def test_prefill_returns_input_length(
        self, mock_torch, mock_model_class, mock_tokenizer_class
    ):
        """prefill returns length of input_ids."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_output = MagicMock()
        mock_model.return_value = mock_output
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        backend = HFBackend()
        backend.load_model("gpt2")
        result = backend.prefill([1, 2, 3, 4, 5])

        assert result == 5


class TestHFBackendForwardOne:
    """Test HFBackend single token forward pass."""

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    @patch("streamagent.engine.backends.hf_backend.torch")
    def test_forward_one_uses_past_key_values(
        self, mock_torch, mock_model_class, mock_tokenizer_class
    ):
        """forward_one uses stored past_key_values."""
        import numpy as np

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096

        # Mock prefill output
        mock_prefill_output = MagicMock()
        mock_past_kv = MagicMock()
        mock_prefill_output.past_key_values = mock_past_kv

        # Mock forward output
        mock_forward_output = MagicMock()
        logits = np.array([[1.0, 2.0, 3.0]])
        mock_forward_output.logits = logits
        mock_forward_output.past_key_values = mock_past_kv

        # Return different outputs for prefill and forward
        mock_model.side_effect = [mock_prefill_output, mock_forward_output]

        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.log_softmax = MagicMock()

        backend = HFBackend()
        backend.load_model("gpt2")
        backend.prefill([1, 2, 3])

        # Verify past_key_values is used in forward_one
        backend.forward_one(token_id=5, cache_position=3)

        # Second call to model should include past_key_values
        assert mock_model.call_count == 2

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    @patch("streamagent.engine.backends.hf_backend.torch")
    def test_forward_one_returns_int_and_float(
        self, mock_torch, mock_model_class, mock_tokenizer_class
    ):
        """forward_one returns (next_token_id: int, log_prob: float)."""
        import numpy as np

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096

        mock_prefill_output = MagicMock()
        mock_prefill_output.past_key_values = MagicMock()

        mock_forward_output = MagicMock()
        logits = np.array([[1.0, 2.0, 3.0]])
        mock_forward_output.logits = logits
        mock_forward_output.past_key_values = MagicMock()

        mock_model.side_effect = [mock_prefill_output, mock_forward_output]

        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        # Mock log_softmax to return valid values
        mock_log_softmax_result = MagicMock()
        mock_log_softmax_result.__getitem__ = MagicMock(return_value=-1.5)
        mock_torch.log_softmax.return_value = mock_log_softmax_result

        backend = HFBackend()
        backend.load_model("gpt2")
        backend.prefill([1, 2, 3])
        next_id, log_prob = backend.forward_one(token_id=5, cache_position=3)

        assert isinstance(next_id, int)
        assert isinstance(log_prob, float)


# ---------------------------------------------------------------------------
# MLXBackend Tests (Stub)
# ---------------------------------------------------------------------------


class TestMLXBackend:
    """Test MLXBackend stub implementation."""

    def test_mlx_load_raises_not_implemented(self):
        """MLXBackend.load_model raises NotImplementedError."""
        backend = MLXBackend()
        with pytest.raises(NotImplementedError):
            backend.load_model("/path/to/model")

    def test_mlx_tokenize_raises_not_implemented(self):
        """MLXBackend.tokenize raises NotImplementedError."""
        backend = MLXBackend()
        with pytest.raises(NotImplementedError):
            backend.tokenize("hello")

    def test_mlx_detokenize_raises_not_implemented(self):
        """MLXBackend.detokenize raises NotImplementedError."""
        backend = MLXBackend()
        with pytest.raises(NotImplementedError):
            backend.detokenize([1, 2, 3])

    def test_mlx_prefill_raises_not_implemented(self):
        """MLXBackend.prefill raises NotImplementedError."""
        backend = MLXBackend()
        with pytest.raises(NotImplementedError):
            backend.prefill([1, 2, 3])

    def test_mlx_forward_one_raises_not_implemented(self):
        """MLXBackend.forward_one raises NotImplementedError."""
        backend = MLXBackend()
        with pytest.raises(NotImplementedError):
            backend.forward_one(token_id=42, cache_position=0)

    def test_mlx_context_length_raises_not_implemented(self):
        """MLXBackend.context_length raises NotImplementedError."""
        backend = MLXBackend()
        with pytest.raises(NotImplementedError):
            _ = backend.context_length


# ---------------------------------------------------------------------------
# Integration & Edge Case Tests
# ---------------------------------------------------------------------------


class TestLlamaBackendIntegration:
    """Integration tests for LlamaBackend workflow."""

    @patch("streamagent.engine.backends.llama_backend.llama_get_logits")
    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_multiple_forward_calls_sequential(self, mock_llama_class, mock_llama_get_logits):
        """Test multiple forward_one calls in sequence."""
        mock_instance = MagicMock()
        mock_instance.n_vocab.return_value = 3
        mock_llama_class.return_value = mock_instance

        # Return different logits for each forward call
        logits1 = [0.5, 1.5, 2.0]
        logits2 = [1.0, 0.5, 1.5]
        mock_llama_get_logits.side_effect = [
            _ctypes_float_array(*logits1),
            _ctypes_float_array(*logits2),
        ]

        backend = LlamaBackend(temperature=0.0)  # Use greedy sampling
        backend.load_model("/path/to/model.gguf", n_ctx=512)
        backend.prefill([1, 2])

        # First forward (uses prefill logits)
        next_id1, log_prob1 = backend.forward_one(token_id=5, cache_position=2)
        assert next_id1 == 2  # argmax of first logits
        assert log_prob1 < 0  # log probability is negative

        # Second forward
        next_id2, log_prob2 = backend.forward_one(token_id=next_id1, cache_position=3)
        assert isinstance(next_id2, int)
        assert log_prob2 < 0

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_empty_tokenize(self, mock_llama_class):
        """tokenize with empty string."""
        mock_instance = MagicMock()
        mock_instance.tokenize.return_value = []
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")
        result = backend.tokenize("")

        assert result == []

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_unicode_text_tokenization(self, mock_llama_class):
        """tokenize with Unicode characters."""
        mock_instance = MagicMock()
        mock_instance.tokenize.return_value = [100, 101]
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf")
        result = backend.tokenize("こんにちは")  # Japanese

        mock_instance.tokenize.assert_called_once_with("こんにちは".encode(), special=True)
        assert isinstance(result, list)

    @patch("streamagent.engine.backends.llama_backend.llama_get_logits")
    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_large_token_ids(self, mock_llama_class, mock_llama_get_logits):
        """prefill and forward with large token IDs."""
        mock_instance = MagicMock()
        mock_instance.n_vocab.return_value = 3
        mock_llama_class.return_value = mock_instance

        logits = [0.1, 0.2, 0.3]
        mock_llama_get_logits.return_value = _ctypes_float_array(*logits)

        backend = LlamaBackend(temperature=0.0)  # Use greedy sampling
        backend.load_model("/path/to/model.gguf")

        large_ids = [30000, 31000, 32000]
        cache_len = backend.prefill(large_ids)
        assert cache_len == 3

        next_id, log_prob = backend.forward_one(token_id=30000, cache_position=3)
        assert isinstance(next_id, int)
        assert isinstance(log_prob, float)

    @patch("streamagent.engine.backends.llama_backend.Llama")
    def test_context_length_large_value(self, mock_llama_class):
        """Test context_length with large value (8K, 32K contexts)."""
        mock_instance = MagicMock()
        mock_llama_class.return_value = mock_instance

        backend = LlamaBackend()
        backend.load_model("/path/to/model.gguf", n_ctx=32768)

        assert backend.context_length == 32768


class TestHFBackendIntegration:
    """Integration tests for HFBackend workflow."""

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    @patch("streamagent.engine.backends.hf_backend.torch")
    def test_multiple_forward_calls_sequential(
        self, mock_torch, mock_model_class, mock_tokenizer_class
    ):
        """Test multiple forward_one calls maintain KV cache correctly."""
        import numpy as np

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096

        # Mock prefill output
        mock_prefill_output = MagicMock()
        mock_past_kv = [MagicMock() for _ in range(24)]  # 24 layers
        mock_prefill_output.past_key_values = mock_past_kv

        # Mock forward outputs
        logits1 = np.array([[1.0, 2.0, 3.0]])
        logits2 = np.array([[2.0, 1.0, 3.0]])
        mock_output1 = MagicMock()
        mock_output1.logits = logits1
        mock_output1.past_key_values = mock_past_kv

        mock_output2 = MagicMock()
        mock_output2.logits = logits2
        mock_output2.past_key_values = mock_past_kv

        mock_model.side_effect = [
            mock_prefill_output,
            mock_output1,
            mock_output2,
        ]

        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        mock_log_softmax_result1 = MagicMock()
        mock_log_softmax_result1.__getitem__ = MagicMock(return_value=-0.5)

        mock_log_softmax_result2 = MagicMock()
        mock_log_softmax_result2.__getitem__ = MagicMock(return_value=-0.3)

        mock_torch.log_softmax.side_effect = [
            mock_log_softmax_result1,
            mock_log_softmax_result2,
        ]

        backend = HFBackend()
        backend.load_model("gpt2")
        backend.prefill([1, 2])

        next_id1, log_prob1 = backend.forward_one(token_id=5, cache_position=2)
        assert isinstance(next_id1, int)
        assert log_prob1 < 0

        next_id2, log_prob2 = backend.forward_one(token_id=next_id1, cache_position=3)
        assert isinstance(next_id2, int)
        assert log_prob2 < 0

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    def test_empty_detokenize(self, mock_model_class, mock_tokenizer_class):
        """detokenize with empty list."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = ""
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HFBackend()
        backend.load_model("gpt2")
        result = backend.detokenize([])

        assert result == ""

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    def test_unicode_detokenize(self, mock_model_class, mock_tokenizer_class):
        """detokenize with unicode output."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "こんにちは"
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 4096
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HFBackend()
        backend.load_model("gpt2")
        result = backend.detokenize([100, 101, 102])

        assert "こんにちは" in result

    @patch("streamagent.engine.backends.hf_backend.AutoTokenizer")
    @patch("streamagent.engine.backends.hf_backend.AutoModelForCausalLM")
    def test_custom_context_length(self, mock_model_class, mock_tokenizer_class):
        """Test overriding context_length with n_ctx kwarg."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.config.max_position_embeddings = 2048
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HFBackend()
        backend.load_model("gpt2", n_ctx=1024)

        # Should use n_ctx override, not model config
        assert backend.context_length == 1024


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


class TestBackendFactory:
    """Test the backend factory function."""

    def test_factory_creates_llama_backend(self):
        """create_backend('llama') returns LlamaBackend instance."""
        backend = create_backend("llama")
        assert isinstance(backend, LlamaBackend)
        assert isinstance(backend, BackendProtocol)

    def test_factory_creates_hf_backend(self):
        """create_backend('hf') returns HFBackend instance."""
        backend = create_backend("hf")
        assert isinstance(backend, HFBackend)
        assert isinstance(backend, BackendProtocol)

    def test_factory_creates_mlx_backend(self):
        """create_backend('mlx') returns MLXBackend instance."""
        backend = create_backend("mlx")
        assert isinstance(backend, MLXBackend)
        assert isinstance(backend, BackendProtocol)

    def test_factory_unknown_backend_raises(self):
        """create_backend with unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("unknown_backend")

    def test_factory_case_sensitive(self):
        """Backend names are case-sensitive."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("Llama")
