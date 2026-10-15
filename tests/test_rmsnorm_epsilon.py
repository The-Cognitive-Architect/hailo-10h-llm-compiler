"""Regression for checkpoint configuration loaded after the exporter import.

Run with the step-1 dependencies installed:
    python -m unittest discover -s tests -p test_rmsnorm_epsilon.py -v
No model download, DFC invocation or accelerator is needed.
"""
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import config
import s1_export_onnx as exporter
import torch
import transformers


class CheckpointEpsilonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_config = {key: value for key, value in vars(config).items() if key.isupper()}
        self.workdir = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {"DFC_WORKDIR": self.workdir.name})
        self.environment.start()
        config.set_workdir(self.workdir.name)
        self.x = torch.tensor([[[0.001, -0.002, 0.003, -0.004]]], dtype=torch.float32)
        self.weight = torch.tensor([1.0, 0.5, 2.0, 1.5], dtype=torch.float32)

    def tearDown(self) -> None:
        for key, value in self.original_config.items():
            setattr(config, key, value)
        self.environment.stop()
        self.workdir.cleanup()

    def load_checkpoint(self, epsilon: float) -> None:
        model = transformers.LlamaConfig(num_attention_heads=2, num_key_value_heads=1,
                                         hidden_size=4, intermediate_size=8, num_hidden_layers=1,
                                         vocab_size=16, rms_norm_eps=epsilon)
        tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=None)
        with patch.object(transformers.AutoConfig, "from_pretrained", return_value=model), \
                patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tokenizer):
            config.load("epsilon-regression-fixture")

    def test_rms_norm_tracks_config_loaded_after_import(self) -> None:
        for epsilon in (1e-5, 1e-4, 1e-6):
            with self.subTest(epsilon=epsilon):
                self.load_checkpoint(epsilon)
                expected = self.x * torch.rsqrt(self.x.square().mean(-1, keepdim=True) + epsilon) * self.weight
                torch.testing.assert_close(exporter.rms_norm(self.x, self.weight), expected, rtol=0, atol=0)

    def test_headwise_rms_norm_tracks_config_loaded_after_import(self) -> None:
        segments = exporter.make_head_seg_matrix(2, 2)
        heads = self.x.reshape(1, 1, 2, 2)
        for epsilon in (1e-5, 1e-4, 1e-6):
            with self.subTest(epsilon=epsilon):
                self.load_checkpoint(epsilon)
                expected = (heads * torch.rsqrt(heads.square().mean(-1, keepdim=True) + epsilon)).reshape_as(self.x)
                expected = expected * self.weight
                actual = exporter.rms_norm_headwise(self.x, self.weight, segments, 2)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_explicit_epsilon_overrides_checkpoint_for_both_helpers(self) -> None:
        self.load_checkpoint(1e-4)
        epsilon = 1e-6
        expected = self.x * torch.rsqrt(self.x.square().mean(-1, keepdim=True) + epsilon) * self.weight
        torch.testing.assert_close(exporter.rms_norm(self.x, self.weight, eps=epsilon), expected, rtol=0, atol=0)
        heads = self.x.reshape(1, 1, 2, 2)
        expected = (heads * torch.rsqrt(heads.square().mean(-1, keepdim=True) + epsilon)).reshape_as(self.x) * self.weight
        actual = exporter.rms_norm_headwise(self.x, self.weight, exporter.make_head_seg_matrix(2, 2), 2, eps=epsilon)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
