import torch

from pcss_single_model.model_v9 import build_model


def test_caro_output_is_ordered_and_bounded():
    model = build_model("caro_rul", channels=43, hidden=8, bins=16).eval()
    with torch.no_grad():
        output = model(torch.randn(3, 43, 32))
    quantiles = output["quantiles"]
    assert quantiles.shape == (3, 4)
    assert torch.all(quantiles[:, 1:] >= quantiles[:, :-1])
    assert torch.all((output["prediction"] >= 0.0) & (output["prediction"] <= 1.0))
