from src.utils.config import Config
from src.utils.common import get_model, get_train_val_datasets, init_wandb_logger
from src.trainer import Trainer


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")

    # --- modello ---
    model = get_model(cfg.get("model_name", "resnet"), num_classes=cfg.get("num_classes", 19))

    # --- dataset (split a livello di image_id, gestito internamente) ---
    train_dataset, val_dataset = get_train_val_datasets(
        csv_path=cfg.csv_path,
        path_data=cfg.path_data,
        path_masks=cfg.path_masks,
        val_split=cfg.get("val_split", 0.2),
        img_size=cfg.get("img_size", 224),
        random_state=cfg.get("random_state", 42),
    )

    # --- logger W&B ---
    logger = init_wandb_logger(
        model=model,
        project=cfg.get("wandb_project", "hpa-single-cell"),
        run_name=cfg.get("wandb_run_name", None),
        config=cfg.__dict__,
    )

    # --- trainer ---
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_path=cfg.output_path,
        logger=logger,
        batch_size=cfg.get("batch_size", 32),
        lr=cfg.get("lr", 1e-4),
        device=cfg.device,
        patience=cfg.get("patience", 5),
        save_path=cfg.get("save_path", "best_model.pth"),
        num_workers=cfg.get("num_workers", 2),
    )

    trainer.fit(epochs=cfg.get("epochs", 30))
