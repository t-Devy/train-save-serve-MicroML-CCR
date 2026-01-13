import json
from pathlib import Path

import torch
import torch.nn as nn

from src.constants import MODEL_PATH, META_PATH, ARTIFACT_DIR
from src.data import make_loaders
from src.model import CCRNet

def evaluate(model: nn.Module, loader, loss_fn, device: torch.device) -> tuple[float, float]:
    """
    Evaluate our model to get our loss value, and accuracy as a tuple of floats

    :param model:
    :param loader:
    :param loss_fn:
    :param device:
    :return avg_loss, accuracy:
    """

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():       # no gradients during evaluation
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            total_loss += loss.item() * xb.size(0)

            # Convert Logits -> probability -> prediction
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            correct += (preds == yb).sum().item()
            total += xb.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

def main():

    #1) Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #2) Data Loaders (CSV -> TensorDataset -> DataLoader)
    train_loader, val_loader, meta = make_loaders(batch_size=16)

    #3) Model
    model = CCRNet(input_dim=len(meta["feature_columns"]), hidden_dim=16).to(device)

    #4) Loss + Optimizer, logits-safe loss, industry safe practice
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    #5) Training Loop
    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

    #6) Save artifacts (files to store model and weights)
    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), str(MODEL_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved meta -> {META_PATH}")

if __name__ == "__main__":
    main()





