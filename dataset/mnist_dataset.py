"""MNIST 커스텀 Dataset 클래스."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    """MNIST 데이터셋 래퍼 클래스.

    Args:
        data_dir: 데이터 저장 경로 (기본값: ./data)
        train: True이면 학습 데이터, False이면 테스트 데이터
        transform: 이미지 변환 (기본값: ToTensor + Normalize)
        visualization: True이면 __getitem__ 호출 시 샘플을 matplotlib으로 표시
    """

    def __init__(
        self,
        data_dir: str = "./data",
        train: bool = True,
        transform=None,
        visualization: bool = False,
    ):
        root = Path(data_dir)
        root.mkdir(parents=True, exist_ok=True)

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

        self.dataset = datasets.MNIST(
            root=str(root), train=train, download=True, transform=transform
        )
        self.visualization = visualization

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]

        if self.visualization:
            import matplotlib.pyplot as plt
            plt.imshow(image.squeeze(0).numpy(), cmap="gray")
            plt.title(f"idx={idx}, label={label}")
            plt.axis("off")
            plt.show()

        return image, label


if __name__ == "__main__":
    dataset = MNISTDataset(visualization=True)
    print(f"데이터셋 크기: {len(dataset)}")

    for image, label in dataset:
        print(f"이미지 shape: {image.shape}, 라벨: {label}")
