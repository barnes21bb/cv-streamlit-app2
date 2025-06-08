import os
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection, VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# simple collate function required for torchvision detection models
def collate_fn(batch):
    return tuple(zip(*batch))


def _load_dataset(data_dir: str, annotation_format: str):
    if annotation_format.lower() == "coco":
        img_dir = os.path.join(data_dir, "images")
        ann_file = os.path.join(data_dir, "annotations.json")
        return CocoDetection(img_dir, ann_file)
    else:
        # assume Pascal VOC style directory
        return VOCDetection(data_dir)


def train_model(
    data_dir: str,
    annotation_format: str = "voc",
    num_classes: int = 2,
    num_epochs: int = 1,
    device: Optional[str] = None,
    metrics_callback: Optional[Callable[[int, dict], None]] = None,
) -> Path:
    """Train a simple Faster R-CNN model on the provided dataset.

    Parameters
    ----------
    data_dir: str
        Dataset directory containing images and annotations.
    annotation_format: str
        Either "voc" or "coco".
    num_classes: int
        Number of classes (including background).
    num_epochs: int
        Training epochs.
    device: str, optional
        Torch device (defaults to CUDA if available).
    metrics_callback: callable(epoch: int, metrics: dict)
        Optional callback executed after each epoch with computed metrics.

    Returns
    -------
    Path to the saved model weights.
    """

    dataset = _load_dataset(data_dir, annotation_format)
    dataset_test = _load_dataset(data_dir, annotation_format)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=2, shuffle=False, collate_fn=collate_fn)

    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    metric = MeanAveragePrecision()

    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(transforms.ToTensor()(img).to(device) if not torch.is_tensor(img) else img.to(device) for img in images)
            targets = [{k: torch.tensor(v).to(device) if isinstance(v, list) else v for k, v in t["annotation"].items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        metric.reset()
        model.eval()
        with torch.no_grad():
            for images, targets in data_loader_test:
                images = list(transforms.ToTensor()(img).to(device) if not torch.is_tensor(img) else img.to(device) for img in images)
                outputs = model(images)
                formatted_targets = [{k: torch.tensor(v) if isinstance(v, list) else v for k, v in t["annotation"].items()} for t in targets]
                metric.update(outputs, formatted_targets)
        stats = metric.compute()
        if metrics_callback:
            metrics_callback(epoch, {k: float(v) for k, v in stats.items()})

    output_path = Path(data_dir) / "trained_model.pth"
    torch.save(model.state_dict(), output_path)
    return output_path


# Snowflake Snowpark Container Services integration (placeholder)
def run_training_in_snowpark(session, image: str, command: str):
    """Run a training job inside Snowpark Container Services.

    This is a lightweight wrapper and requires an existing Snowflake session.
    """
    try:
        from snowflake.snowpark.container import Container
    except Exception:  # pragma: no cover - library might be missing in tests
        raise RuntimeError("Snowpark Container Services package not available")

    container = Container(image=image, command=command)
    result = container.run(session)
    return result


def upload_to_huggingface(model_path: str, repo_id: str, token: str):
    """Upload a trained model to Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_id,
        token=token,
    )

