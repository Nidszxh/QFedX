import torch
import numpy as np
from typing import Dict, List, Tuple

def apply_dp_noise(delta_dict: Dict[str, torch.Tensor], clip_norm: float, noise_multiplier: float) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Apply Local Differential Privacy (LDP) noise to model updates (delta).
    
    Args:
        delta_dict: Dictionary of parameter updates (\Delta \theta).
        clip_norm: Maximum L2 norm C for clipping.
        noise_multiplier: Noise multiplier \sigma for Gaussian noise.
        
    Returns:
        (dp_delta_dict, actual_norm)
    """
    dp_delta = {}
    
    # 1. Compute L2 norm of the entire parameter update
    sq_norm = 0.0
    for k, v in delta_dict.items():
        sq_norm += torch.sum(v.float() ** 2).item()
    actual_norm = np.sqrt(sq_norm)
    
    # 2. Compute clipping factor (max 1.0)
    clip_scale = min(1.0, clip_norm / (actual_norm + 1e-10))
    
    # 3. Clip and add noise
    for k, v in delta_dict.items():
        clipped_v = v.float() * clip_scale
        
        if noise_multiplier > 0:
            # Noise scale is noise_multiplier * clip_norm
            noise = torch.randn_like(clipped_v) * (noise_multiplier * clip_norm)
        else:
            noise = torch.zeros_like(clipped_v)
            
        dp_delta[k] = (clipped_v + noise).to(v.dtype)
        
    return dp_delta, actual_norm

class SecureAggregator:
    """
    Simulates Pairwise Secure Aggregation mapping.
    Uses shared PRG seeds between client pairs to generate additive masks.
    """
    def __init__(self, num_clients: int, client_ids: List[int], parameter_template: Dict[str, torch.Tensor]):
        self.num_clients = num_clients
        self.client_ids = client_ids
        self.parameter_template = parameter_template
        
        # Simulate Diffie-Hellman seed exchange
        # seed_matrix[i][j] gives the shared seed between client i and j
        self.seed_matrix = np.random.randint(0, 1000000, size=(num_clients, num_clients))
        # Ensure symmetry
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                self.seed_matrix[j][i] = self.seed_matrix[i][j]
                
    def get_client_mask(self, client_idx: int) -> Dict[str, torch.Tensor]:
        """
        Generates the aggregate mask for a specific client.
        m_i = \sum_{j < i} PRG(s_ij) - \sum_{j > i} PRG(s_ij)
        """
        mask = {k: torch.zeros_like(v).float() for k, v in self.parameter_template.items()}
        
        # Only active clients participate in the current round mask
        for peer_idx in self.client_ids:
            if peer_idx == client_idx:
                continue
                
            # Seed shared with peer
            shared_seed = self.seed_matrix[client_idx][peer_idx]
            
            # Generate deterministic noise state from seed
            generator = torch.Generator(device=next(iter(mask.values())).device)
            generator.manual_seed(int(shared_seed))
            
            # Add or subtract depending on ID ordering
            sign = 1 if client_idx > peer_idx else -1
            
            for k in mask:
                shape = mask[k].shape
                # Generate random tensor with fixed seed
                peer_mask = torch.randn(shape, generator=generator, dtype=torch.float32, device=mask[k].device)
                mask[k] += sign * peer_mask
                
        return mask

if __name__ == "__main__":
    print("Testing Privacy Primitives...")
    
    # 1. Test DP Mechanism
    template = {'w1': torch.ones(2, 2) * 5.0, 'w2': torch.ones(3) * 10}
    clipped, norm = apply_dp_noise(template, clip_norm=5.0, noise_multiplier=0.0)
    print(f"Original norm: {norm:.2f}")
    
    # Check max norm is exactly 5.0
    post_norm = np.sqrt(sum(torch.sum(v**2).item() for v in clipped.values()))
    print(f"Post-clipping norm: {post_norm:.2f}")
    assert np.isclose(post_norm, 5.0), "Clipping failed to bound L2 norm"
    
    # 2. Test Secure Aggregation Math Cancellation
    print("\nTesting Secure Aggregation Mask Cancellation...")
    client_ids = [0, 1, 2, 3]
    aggregator = SecureAggregator(num_clients=4, client_ids=client_ids, parameter_template=template)
    
    total_mask = {k: torch.zeros_like(v).float() for k, v in template.items()}
    
    for cid in client_ids:
        client_mask = aggregator.get_client_mask(cid)
        for k in total_mask:
            total_mask[k] += client_mask[k]
            
    # Verify absolute sums are zeroes
    for k, v in total_mask.items():
        max_val = torch.max(torch.abs(v)).item()
        print(f"Max residual for {k}: {max_val:.6f}")
        assert max_val < 1e-4, f"Mask cancellation failed for {k}! Residual: {max_val}"
        
    print("✅ All privacy tests passed.")
