import numpy as np
import torch
from Preprocess import preprocess_mnist
from qAmplitude import amplitude_encode, get_statevector_from_circuit
from qAngle import angle_encode
import random
import matplotlib.pyplot as plt


"""
Workflow:
1. Preprocess MNIST (digits 0,1,2) using preprocess_mnist.
2. Take one training sample.
3. Downsample to 4x4 (16 values).
4. Apply amplitude encoding (16 → 4 qubits).
5. Apply angle encoding (pool 16 → 4 features → 4 qubits).
6. Print/visualize both circuits.
"""

def downsample_image(img: np.ndarray, out_shape=(4, 4)) -> np.ndarray:
    """Downsample a 2D image (H x W) to out_shape using simple block averaging."""
    img = np.asarray(img, dtype=float)
    in_h, in_w = img.shape
    out_h, out_w = out_shape
    h_step = in_h / out_h
    w_step = in_w / out_w
    out = np.zeros(out_shape)
    for i in range(out_h):
        for j in range(out_w):
            h0 = int(np.floor(i * h_step))
            h1 = int(np.floor((i + 1) * h_step))
            w0 = int(np.floor(j * w_step))
            w1 = int(np.floor((j + 1) * w_step))
            if h1 <= h0:
                h1 = min(in_h, h0 + 1)
            if w1 <= w0:
                w1 = min(in_w, w0 + 1)
            patch = img[h0:h1, w0:w1]
            out[i, j] = patch.mean() if patch.size > 0 else 0.0
    return out

def pool_to_n_features(vec: np.ndarray, n_features: int) -> np.ndarray:
    """Pool a 1D vector into n_features by averaging contiguous chunks."""
    v = np.asarray(vec, dtype=float)
    L = v.size
    if n_features >= L:
        out = np.zeros(n_features)
        out[:L] = v
        return out
    chunk_size = L // n_features
    out = np.zeros(n_features)
    for i in range(n_features):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_features - 1 else L
        out[i] = v[start:end].mean()
    return out

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    config = {
        'raw_folder': "./dataset/raw",
        'processed_folder': "./dataset/processed",
        'digits': (0, 1, 2),
        'val_split': 0.1,
        'num_clients': 4,
        'partition_type': 'iid',  # or 'non_iid'
        'alpha': 0.5
    }

    print("Preprocessing MNIST dataset...")
    result = preprocess_mnist(**config)

    if result is None:
        raise RuntimeError(
            "Preprocessing failed. Please check dataset files in ./dataset/raw "
            "(expected: train-images.idx3-ubyte, train-labels.idx1-ubyte, "
            "t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte)"
        )

    train_data, val_data, test_data, client_data = result

    # Extract one sample (first image of training set)
    X_train, y_train = train_data
    img = X_train[0].squeeze().numpy()  # shape (28,28)
    label = y_train[0].item()

    print(f"Using sample digit with label: {label}")

    # Downsample to 4x4
    down = downsample_image(img, out_shape=(4, 4))
    flat16 = down.flatten()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original 28x28")
    axes[0].axis('off')

    axes[1].imshow(down, cmap='gray', interpolation='nearest')
    axes[1].set_title("Downsampled 4x4")
    axes[1].axis('off')

    plt.suptitle(f"Digit: {label}")
    plt.tight_layout()
    plt.show()

    print("Downsampled 4x4 image:\n", down)

    # -------- Amplitude Encoding --------
    amp_qc = amplitude_encode(flat16)
    print('\nAmplitude encoding circuit:')
    print(amp_qc.draw(output='text'))

    sv = get_statevector_from_circuit(amp_qc)
    print('\nAmplitude statevector (first 8 amplitudes):')
    np.set_printoptions(precision=4, suppress=True)
    print(sv.data[:8])

    # -------- Angle Encoding --------
    features4 = pool_to_n_features(flat16, 4)
    ang_qc = angle_encode(features4, n_qubits=4, basis='ry')
    print('\nAngle encoding circuit:')
    print(ang_qc.draw(output='text'))

    print("\nDone: Both amplitude and angle encoding circuits created.")
