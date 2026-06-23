import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_path: str,
        logger: wandb.sdk.wandb_run.Run,
        batch_size: int = 32,
        lr: float = 1e-4,
        device: str = "cuda",
        patience: int = 5,
        save_path: str = "best_model.pth",
        num_workers: int = 2,
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.patience = patience
        self.save_path = save_path
        self.output_path = output_path
        self.logger = logger

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def train_epoch(self) -> float:
        self.model.train()
        total_loss: float = 0.0
        for crops, labels in self.train_loader:
            crops = crops.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs: torch.Tensor = self.model(crops)
            loss: torch.Tensor = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def val_epoch(self) -> float:
        self.model.eval()
        total_loss: float = 0.0
        with torch.no_grad():
            for crops, labels in self.val_loader:
                crops = crops.to(self.device)
                labels = labels.to(self.device)
                outputs: torch.Tensor = self.model(crops)
                loss: torch.Tensor = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def fit(self, epochs: int) -> None:
        best_val_loss: float = float("inf")
        epochs_without_improvement: int = 0

        for epoch in range(epochs):
            train_loss: float = self.train_epoch()
            val_loss: float = self.val_epoch()

            print(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}", end="")

            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(" ← best model saved", end="")

                artifact = wandb.Artifact(
                    name=f"{self.logger.name}-best-model",
                    type="model",
                    metadata={"epoch": epoch + 1, "val_loss": val_loss},
                )
                artifact.add_file(self.save_path)
                self.logger.log_artifact(artifact)
            else:
                epochs_without_improvement += 1
                print(f" (no improvement {epochs_without_improvement}/{self.patience})", end="")

            log_dict["best_val_loss"] = best_val_loss
            self.logger.log(log_dict)
            print()

            if epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1} — best val_loss: {best_val_loss:.4f}")
                break

        self.model.load_state_dict(torch.load(self.save_path))
        print("Best model loaded.")
        self.logger.finish()