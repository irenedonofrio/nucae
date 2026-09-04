"""Parity gate: nucae.unet reproduces the cluster networks exactly.

The three files in ../old_nucae are the implementations that produced every
existing checkpoint. This test is the reason nucae/unet.py may be trusted as a
clean rewrite rather than a new model: for each variant it asserts that the
state_dict keys and shapes are identical, that the old weights load with
strict=True, and that both networks then return bit-identical output.

FAILS, not skips, when old_nucae/ is absent. A skip here would leave a green
suite that verified nothing, on the one test the README calls the reason this
rewrite may be trusted. Absent reference means unverified port, and unverified
must not look like passed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from nucae.unet import FragmentomicsUNet

OLD = Path(__file__).parents[3] / "old_nucae"

# (id, old module, old kwargs, new kwargs, input channels, parameter count)
# Counts are torchinfo's, as recorded in the thesis draft. V1 and V2-with-sequence
# are the same network by construction and so share a count.
VARIANTS = [
    ("v1", "model_v1", {}, {"use_sequence": True}, 6, 8_517_953),
    ("v2_seq", "model_v2", {"use_sequence": True}, {"use_sequence": True}, 6, 8_517_953),
    ("v2_noseq", "model_v2", {"use_sequence": False}, {"use_sequence": False}, 1, 8_515_393),
    ("v3", "model_v3", {}, {"use_sequence": True, "dual_stream": True}, 6, 8_545_665),
]

LENGTH = 1024          # divisible by 8, the encoder's total stride; 25,000 is only slower
SEED = 0


def reference_net(module_name: str, kwargs: dict) -> torch.nn.Module:
    """Import old_nucae/<module_name>.py by path and build its bare UNet.

    Loaded under a unique module name because all three files define a class
    called FragmentomicsUNet and would otherwise shadow one another.
    """
    path = OLD / f"{module_name}.py"
    if not path.exists():
        pytest.fail(
            f"reference implementation not found: {path}\n"
            "\n"
            "This test is what proves nucae/unet.py reproduces the networks that\n"
            "produced every existing checkpoint. Without the reference, nothing\n"
            "in this suite verifies the port.\n"
            "\n"
            "Check out old_nucae/ as a sibling of this repository and re-run."
        )
    spec = importlib.util.spec_from_file_location(f"_old_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FragmentomicsUNet(**kwargs)


@pytest.fixture(params=VARIANTS, ids=[v[0] for v in VARIANTS])
def variant(request):
    """One variant: the reference net, the rewritten net, and what to expect."""
    _, module_name, old_kwargs, new_kwargs, in_channels, n_params = request.param
    old = reference_net(module_name, old_kwargs)
    new = FragmentomicsUNet(**new_kwargs)
    return old, new, in_channels, n_params


def test_state_dict_keys_and_shapes_match(variant):
    """Same names in the same order, same shapes -- what makes old .ckpt loadable."""
    old, new, _, _ = variant
    old_sd, new_sd = old.state_dict(), new.state_dict()
    assert list(new_sd) == list(old_sd)
    for key in old_sd:
        assert new_sd[key].shape == old_sd[key].shape, key


def test_old_weights_load_strict(variant):
    """strict=True: no missing keys, no unexpected ones."""
    old, new, _, _ = variant
    new.load_state_dict(old.state_dict(), strict=True)


def test_forward_is_bit_identical(variant):
    """Identical weights and identical input must give identical output, exactly.

    torch.equal, not allclose: the two are the same operations in the same order,
    so any difference is a structural divergence, not float noise.
    """
    old, new, in_channels, _ = variant
    new.load_state_dict(old.state_dict(), strict=True)
    old.eval()
    new.eval()

    generator = torch.Generator().manual_seed(SEED)
    x = torch.randn(2, in_channels, LENGTH, generator=generator)

    with torch.no_grad():
        assert torch.equal(new(x), old(x))


def test_parameter_count(variant):
    """The only thing distinguishing variants in a checkpoint."""
    _, new, _, n_params = variant
    assert sum(p.numel() for p in new.parameters()) == n_params


def test_output_shape_is_one_channel_at_input_length(variant):
    _, new, in_channels, _ = variant
    with torch.no_grad():
        assert new(torch.zeros(1, in_channels, LENGTH)).shape == (1, 1, LENGTH)


def test_dual_stream_without_sequence_is_refused():
    """V3 reads channels 1-5; there is nothing coherent to do without them."""
    with pytest.raises(ValueError):
        FragmentomicsUNet(use_sequence=False, dual_stream=True)
