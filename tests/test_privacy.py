import pytest
import torch

from quantum.privacy.differential_privacy import (
    DifferentialPrivacy,
    add_gaussian_noise,
    clip_gradients,
)
from quantum.privacy.privacy_accountant import PrivacyAccountant
from quantum.privacy.secure_aggregation import SecureAggregator


class TestClipGradients:
    def test_clip_under_threshold(self):
        grads = {"w": torch.tensor([0.1, 0.2, 0.3])}
        clipped, norm = clip_gradients(grads, clip_norm=1.0)
        assert torch.allclose(clipped["w"], grads["w"])
        assert abs(norm - 0.3742) < 1e-3

    def test_clip_over_threshold(self):
        grads = {"w": torch.tensor([3.0, 4.0])}
        clipped, norm = clip_gradients(grads, clip_norm=1.0)
        expected_norm = 5.0
        assert abs(norm - expected_norm) < 1e-4
        assert torch.allclose(clipped["w"], grads["w"] / 5.0)

    def test_multiple_keys(self):
        grads = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([2.0, 0.0])}
        clipped, norm = clip_gradients(grads, clip_norm=3.0)
        assert norm > 0
        assert clipped["a"].shape == (2,)

    def test_empty_grads(self):
        clipped, norm = clip_gradients({}, clip_norm=1.0)
        assert clipped == {}
        assert norm == 0.0

    def test_none_gradients(self):
        grads = {"w": None, "b": torch.tensor([1.0])}
        clipped, norm = clip_gradients(grads, clip_norm=1.0)
        assert "w" not in clipped
        assert "b" in clipped


class TestAddGaussianNoise:
    def test_noise_shape_matches(self):
        grads = {"w": torch.zeros(5, 5)}
        noisy = add_gaussian_noise(grads, noise_scale=0.1)
        assert noisy["w"].shape == (5, 5)

    def test_noise_is_nonzero(self):
        grads = {"w": torch.zeros(100)}
        noisy = add_gaussian_noise(grads, noise_scale=1.0)
        assert noisy["w"].std() > 0.5

    def test_zero_noise_scale(self):
        grads = {"w": torch.tensor([1.0, 2.0])}
        noisy = add_gaussian_noise(grads, noise_scale=0.0)
        assert torch.allclose(noisy["w"], grads["w"])

    def test_multiple_keys(self):
        grads = {"a": torch.zeros(3), "b": torch.ones(2)}
        noisy = add_gaussian_noise(grads, noise_scale=0.5)
        assert set(noisy.keys()) == {"a", "b"}


class TestDifferentialPrivacy:
    def test_apply_no_gradients(self):
        dp = DifferentialPrivacy(clip_norm=1.0, noise_multiplier=1.0)
        params = {"w": torch.zeros(3, requires_grad=False)}
        result, norm = dp.apply(params, batch_size=1, sample_rate=1.0)
        assert norm == 0.0

    def test_apply_with_gradients(self):
        dp = DifferentialPrivacy(clip_norm=1.0, noise_multiplier=0.0)
        w = torch.tensor([3.0, 4.0], requires_grad=True)
        loss = w.sum()
        loss.backward()
        params = {"w": w}
        result, norm = dp.apply(params, batch_size=1, sample_rate=1.0)
        expected_norm = (1.0**2 + 1.0**2) ** 0.5
        assert abs(norm - expected_norm) < 1e-4
        assert result["w"].grad is not None

    def test_state_dict(self):
        dp = DifferentialPrivacy(clip_norm=2.0, noise_multiplier=0.5)
        sd = dp.state_dict()
        assert sd["clip_norm"] == 2.0
        assert sd["noise_multiplier"] == 0.5

    def test_load_state_dict(self):
        dp = DifferentialPrivacy()
        dp.load_state_dict({"clip_norm": 3.0, "noise_multiplier": 0.1})
        assert dp.clip_norm == 3.0
        assert dp.noise_multiplier == 0.1


class TestPrivacyAccountant:
    def test_compute_eps_gaussian(self):
        pa = PrivacyAccountant(delta=1e-5)
        eps = pa.compute_eps_gaussian(q=0.1, sigma=2.0, steps=100)
        assert eps > 0
        assert eps < float('inf')

    def test_compute_eps_gaussian_q_ge_1(self):
        pa = PrivacyAccountant(delta=1e-5)
        eps = pa.compute_eps_gaussian(q=0.1, sigma=2.0, steps=100)
        assert eps > 0

    def test_zero_sigma(self):
        pa = PrivacyAccountant()
        assert pa.compute_eps_gaussian(q=0.1, sigma=0, steps=10) == float('inf')
        assert pa.compute_eps_gaussian(q=0.1, sigma=0, steps=10) == float('inf')

    def test_step_and_get(self):
        pa = PrivacyAccountant(delta=1e-5)
        eps_step = pa.step(q=0.1, sigma=2.0)
        assert eps_step > 0
        assert pa.get_privacy_spent() == pytest.approx(eps_step)

    def test_multiple_steps(self):
        pa = PrivacyAccountant(delta=1e-5)
        for _ in range(10):
            pa.step(q=0.1, sigma=2.0)
        assert pa.get_privacy_spent() > 0

    def test_reset(self):
        pa = PrivacyAccountant(delta=1e-5)
        pa.step(q=0.1, sigma=2.0)
        pa.reset()
        assert pa.get_privacy_spent() == 0.0
        assert pa.q_cumulative == 0.0

    def test_state_dict(self):
        pa = PrivacyAccountant(delta=1e-5)
        pa.step(q=0.1, sigma=2.0)
        sd = pa.state_dict()
        assert "delta" in sd
        assert "eps_cumulative" in sd
        assert sd["delta"] == 1e-5

    def test_load_state_dict(self):
        pa = PrivacyAccountant()
        pa.load_state_dict({"delta": 1e-3, "eps_cumulative": 5.0, "q_cumulative": 2.0})
        assert pa.delta == 1e-3
        assert pa.eps_cumulative == 5.0
        assert pa.q_cumulative == 2.0


class TestSecureAggregator:
    def test_generate_masks_count(self):
        agg = SecureAggregator(num_clients=3)
        shapes = {"w": torch.Size([4, 4])}
        masks = agg.generate_masks_for_client(0, shapes)
        assert "w" in masks
        assert masks["w"].shape == (4, 4)

    def test_masks_cancel_across_clients(self):
        agg = SecureAggregator(num_clients=3, modulus=1e3)
        shapes = {"w": torch.Size([3])}
        all_masks = []
        for i in range(3):
            m = agg.generate_masks_for_client(i, shapes)
            all_masks.append(m)
        summed = torch.zeros(3)
        for m in all_masks:
            summed += m["w"]
        assert torch.allclose(summed, torch.zeros(3), atol=1e-2)

    def test_mask_and_unmask(self):
        agg = SecureAggregator(num_clients=3, modulus=1e3)
        shapes = {"w": torch.Size([4])}
        base_params = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        all_masks = []
        for i in range(3):
            m = agg.generate_masks_for_client(i, shapes, dtype=torch.float64)
            all_masks.append(m)
        masked_params = {"w": torch.zeros(4, dtype=torch.float64)}
        for i in range(3):
            client_params = {"w": base_params.clone()}
            masked = agg.mask_params(client_params, all_masks[i])
            masked_params["w"] = masked_params["w"] + masked["w"]
        result = agg.unmask_params(masked_params, all_masks)
        expected = base_params * 3
        assert torch.allclose(result["w"], expected, atol=1e-3)

    def test_empty_masks_unmask(self):
        agg = SecureAggregator(num_clients=3)
        params = {"w": torch.tensor([1.0, 2.0])}
        result = agg.unmask_params(params, [])
        assert result["w"] is params["w"]

    def test_state_dict(self):
        agg = SecureAggregator(num_clients=5)
        sd = agg.state_dict()
        assert sd["num_clients"] == 5

    def test_deterministic_masks(self):
        agg = SecureAggregator(num_clients=3)

        torch.manual_seed(42)
        shapes = {"w": torch.Size([2, 2])}
        m1 = agg.generate_masks_for_client(0, shapes, dtype=torch.float64)
        torch.manual_seed(42)
        m2 = agg.generate_masks_for_client(0, shapes, dtype=torch.float64)
        assert torch.allclose(m1["w"], m2["w"])
