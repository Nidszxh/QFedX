from pathlib import Path

import numpy as np
import pytest

MNIST_RAW = Path(__file__).resolve().parent.parent / 'dataset' / 'raw'


class TestPreprocessUtils:
    @pytest.mark.skipif(not (MNIST_RAW / 'train-images.idx3-ubyte').exists(),
                        reason="MNIST raw data not found")
    def test_read_idx_images(self):
        from data.preprocess import read_idx_images
        data = read_idx_images(str(MNIST_RAW / 'train-images.idx3-ubyte'))
        assert data.shape[0] > 0
        assert data.ndim == 3

    @pytest.mark.skipif(not (MNIST_RAW / 'train-labels.idx1-ubyte').exists(),
                        reason="MNIST raw data not found")
    def test_read_idx_labels(self):
        from data.preprocess import read_idx_labels
        data = read_idx_labels(str(MNIST_RAW / 'train-labels.idx1-ubyte'))
        assert data.ndim == 1

    def test_create_iid_partition(self):
        from data.preprocess import create_iid_partition
        rng = np.random.default_rng(42)
        indices = np.arange(100)
        partitions = create_iid_partition(indices, 4, rng)
        assert len(partitions) == 4
        total = sum(len(p) for p in partitions)
        assert total == 100

    def test_create_non_iid_partition(self):
        from data.preprocess import create_non_iid_partition
        rng = np.random.default_rng(42)
        y = np.repeat(np.arange(3), 30)
        partitions = create_non_iid_partition(y, 3, alpha=0.5, rng=rng)
        assert len(partitions) == 3

    def test_create_partition_iid(self):
        from data.preprocess import create_partition
        y = np.arange(100)
        partitions = create_partition(y, 4, alpha=None, seed=42)
        assert len(partitions) == 4

    def test_create_partition_non_iid(self):
        from data.preprocess import create_partition
        y = np.repeat(np.arange(3), 30)
        partitions = create_partition(y, 3, alpha=0.5, seed=42)
        assert len(partitions) == 3


class TestVizPreprocess:
    def test_generate_all_visualizations(self, tmp_path):
        from data.plots_preprocess import generate_all_preprocessing_visualizations
        save_dir = str(tmp_path / 'viz')
        generate_all_preprocessing_visualizations(
            pca_model=None,
            data_before_scaling=np.random.randn(100, 4),
            data_after_scaling=np.random.randn(100, 4),
            client_indices=[np.arange(25), np.arange(25)],
            save_dir=save_dir,
        )
        assert Path(save_dir).exists()
