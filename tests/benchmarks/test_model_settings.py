from contextlib import contextmanager

import pytest

from benchmarks.model_settings import stable_local_model_kwargs


@pytest.mark.parametrize("device,expected", [("cuda:2", "cuda:2"), (None, 1), ("cpu", None)])
def test_precision_probe_uses_the_selected_device(monkeypatch, device, expected):
    import torch

    active = []
    probed = []

    @contextmanager
    def device_context(selected):
        active.append(selected)
        try:
            yield
        finally:
            active.pop()

    def supports_bf16():
        assert active
        probed.append(active[-1])
        return True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supports_bf16)
    result = stable_local_model_kwargs(device)
    assert result == {"torch_dtype": torch.float32 if expected is None else torch.bfloat16}
    assert probed == ([] if expected is None else [expected])
    assert active == []
