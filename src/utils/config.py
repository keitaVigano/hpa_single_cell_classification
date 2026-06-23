from pathlib import Path
import torch
import yaml


class Config:

    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        if self.__dict__:
            return
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if raw.get("device") == "cuda" and not torch.cuda.is_available():
            raw["device"] = "cpu"

        return cls(**raw)

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({fields})"

    def get(self, key: str, default=None):
        """Accesso sicuro: non esplode se la chiave non esiste."""
        return getattr(self, key, default)