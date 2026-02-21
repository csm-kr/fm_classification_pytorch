"""MNIST DataLoader 빌더."""

from __future__ import annotations

from torch.utils.data import DataLoader

from mnist_dataset import MNISTDataset


def build_dataloader(
    data_dir: str = "./data",
    batch_size: int = 64,
    test_batch_size: int = 1000,
    num_workers: int = 0,
    visualization: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """학습/테스트 DataLoader를 생성해 반환한다.

    Args:
        data_dir: 데이터 저장 경로
        batch_size: 학습 배치 크기
        test_batch_size: 테스트 배치 크기
        num_workers: DataLoader 워커 수
        visualization: True이면 샘플 접근 시 matplotlib으로 표시

    Returns:
        (train_loader, test_loader) 튜플
    """
    train_dataset = MNISTDataset(data_dir=data_dir, train=True, visualization=visualization)
    test_dataset = MNISTDataset(data_dir=data_dir, train=False, visualization=visualization)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = build_dataloader()
    images, labels = next(iter(train_loader))
    print(f"학습 배치 이미지 shape: {tuple(images.shape)}")
    print(f"학습 배치 라벨 shape: {tuple(labels.shape)}")
    print(f"학습 배치 수: {len(train_loader)}, 테스트 배치 수: {len(test_loader)}")
