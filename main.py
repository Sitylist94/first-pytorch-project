import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)
n_samples = 1000

X = torch.randn(n_samples, 3)

noise = 0.2 * torch.randn(n_samples, 1)

y = (
    3 * X[:, 0:1]
    - 2 * X[:, 1:2]
    + 0.5 * X[:, 2:3]
    + 4
    + noise
)

n_train = 700
n_val = 150

indices = torch.randperm(n_samples)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

mean = X_train.mean(dim=0)
std = X_train.std(dim=0).clamp_min(1e-8)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = RegressionModel().to(device)

print(model)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

num_epochs = 100

for epoch in range(num_epochs):
    # MODE ENTRAÎNEMENT
    model.train()

    train_loss_sum = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 1. Prédiction
        predictions = model(X_batch)

        # 2. Calcul de la loss
        loss = loss_fn(predictions, y_batch)

        # 3. Effacer les anciens gradients
        optimizer.zero_grad()

        # 4. Calculer les gradients
        loss.backward()

        # 5. Mettre à jour les paramètres
        optimizer.step()

        train_loss_sum += loss.item() * len(X_batch)

    train_loss = train_loss_sum / len(train_dataset)

    # MODE VALIDATION
    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)

            val_loss_sum += loss.item() * len(X_batch)

    val_loss = val_loss_sum / len(val_dataset)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Validation: {val_loss:.4f}"
        )

model.eval()

test_loss_sum = 0.0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)

        test_loss_sum += loss.item() * len(X_batch)

test_mse = test_loss_sum / len(test_dataset)

print(f"Test MSE : {test_mse:.4f}")
print(f"Test RMSE : {test_mse ** 0.5:.4f}")

new_X = torch.tensor([[0.5, -1.0, 2.0]])

new_X_normalized = (new_X - mean) / std
new_X_normalized = new_X_normalized.to(device)

model.eval()

with torch.no_grad():
    prediction = model(new_X_normalized)

print("Prédiction :", prediction.item())