"""Unit tests for the torch <-> jax DLPack bridge."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from factory_jax.bridge.tensor_convert import from_jax, to_jax


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_to_jax_torch_cuda_roundtrip_values():
    x_torch = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    a = to_jax(x_torch)
    assert isinstance(a, jax.Array)
    assert a.shape == (8, 16)
    assert "cuda" in str(a.devices()).lower(), f"expected cuda device, got {a.devices()}"
    np.testing.assert_allclose(np.asarray(a), x_torch.cpu().numpy(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_from_jax_to_torch_values():
    a = jnp.arange(32, dtype=jnp.float32).reshape(4, 8)
    a = jax.device_put(a, jax.devices()[0])
    x = from_jax(a, target="torch")
    assert isinstance(x, torch.Tensor)
    assert x.is_cuda
    assert x.shape == (4, 8)
    np.testing.assert_allclose(x.cpu().numpy(), np.asarray(a), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_roundtrip_torch_jax_torch_exact():
    x_in = torch.randn(64, device="cuda", dtype=torch.float32)
    x_out = from_jax(to_jax(x_in), target="torch")
    np.testing.assert_array_equal(x_in.cpu().numpy(), x_out.cpu().numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_to_jax_handles_noncontiguous_torch():
    """Verify the .contiguous() guard inside to_jax."""
    x_full = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    x_slice = x_full[:, ::2]  # noncontiguous stride
    assert not x_slice.is_contiguous()
    a = to_jax(x_slice)
    assert a.shape == (8, 8)
    np.testing.assert_allclose(np.asarray(a), x_slice.contiguous().cpu().numpy(), rtol=0, atol=0)


def test_to_jax_rejects_unsupported_source():
    with pytest.raises(TypeError, match="unsupported source type"):
        to_jax([1, 2, 3])
