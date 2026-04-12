from preprocess import preprocess_mnist_classical, preprocess_mnist_quantum

print("Testing Classical Mode...")
train_c, val_c, test_c, clients_c = preprocess_mnist_classical(
    digits=(0, 1, 2),
    num_clients=4,
    seed=42,
    generate_plots=False
)

# Verify classical specs
assert clients_c[0][0].shape[1] == 784, "Should be 784D"
assert 0 <= clients_c[0][0].min() and clients_c[0][0].max() <= 1, "Should be [0,1]"
print(f"✅ Classical: {clients_c[0][0].shape[1]}D, range [{clients_c[0][0].min():.3f}, {clients_c[0][0].max():.3f}]")

print("\nTesting Quantum Mode...")
train_q, val_q, test_q, clients_q = preprocess_mnist_quantum(
    digits=(0, 1, 2),
    num_clients=4,
    pca_components=4,
    seed=42,
    generate_plots=False
)

# Verify quantum specs
assert clients_q[0][0].shape[1] == 4, "Should be 4D"
assert -1 <= clients_q[0][0].min() and clients_q[0][0].max() <= 1, "Should be [-1,1]"
print(f"✅ Quantum: {clients_q[0][0].shape[1]}D, range [{clients_q[0][0].min():.3f}, {clients_q[0][0].max():.3f}]")

print("\n🎉 Both modes work perfectly!")