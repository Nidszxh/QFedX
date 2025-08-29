import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from qAmplitude import amplitude_encode   
from qAngle import angle_encode           
import pennylane as qml

# Add parent folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.Preprocess import preprocess_mnist 

# QNN parameters
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

def variational_layer(params):
    for i in range(n_qubits):
        qml.RX(params[i, 0], wires=i)
        qml.RY(params[i, 1], wires=i)
        qml.RZ(params[i, 2], wires=i)
    # Simple entangling layer
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev, interface='torch')
def qnode(x, params, encoding='amplitude'):
    if encoding == 'amplitude':
        amplitude_encode(x)
    elif encoding == 'angle':
        angle_encode(x, n_qubits)
    else:
        raise ValueError("Encoding must be 'amplitude' or 'angle'")
    
    variational_layer(params)
    
    # Return as tensor instead of list
    return torch.tensor([qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])


class QNN(nn.Module):
    def __init__(self, n_qubits, n_classes=3, encoding='amplitude'):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.params = nn.Parameter(0.01 * torch.randn(n_qubits, 3))
        self.fc = nn.Linear(n_qubits, n_classes)
    
    def forward(self, x):
        # Apply QNode to each sample
        q_out = torch.stack([qnode(xi, self.params, self.encoding) for xi in x])
        logits = self.fc(q_out)
        return logits

if __name__ == "__main__":
    # Preprocess MNIST digits {0,1,2} with PCA reduced to n_qubits
    train_set, val_set, test_set, client_data = preprocess_mnist(
        raw_folder="./dataset/raw",
        processed_folder="./dataset/processed",
        digits=(0, 1, 2),
        val_split=0.1,
        num_clients=4,
        partition_type='iid',
        alpha=0.5,
        apply_pca=True,
        pca_components=n_qubits
    )

    X_train, y_train = train_set

    # Convert to torch tensors properly
    X_train = X_train.clone().detach() if isinstance(X_train, torch.Tensor) else torch.tensor(X_train, dtype=torch.float32)
    y_train = y_train.clone().detach() if isinstance(y_train, torch.Tensor) else torch.tensor(y_train, dtype=torch.long)

    # Initialize QNN
    model = QNN(n_qubits=n_qubits, n_classes=3, encoding='amplitude')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
