# MNIST 분류 프로젝트 계획

## 프로젝트 목표

MNIST 손글씨 숫자 분류를 위한 전체 AI 파이프라인 구축.
데이터 준비 → 모델 정의 → 학습 → 평가까지 단계별로 구조화한다.

---

## 폴더 구조

```
mnist_classifiation_fm/
├── dataset/
│   ├── mnist_dataset.py      # MNISTDataset 클래스
│   └── mnist_dataloader.py   # build_dataloader() 함수
├── model/                    # TBD
├── train/                    # TBD
├── eval/                     # TBD
├── data/                     # MNIST 원본 데이터 (자동 다운로드)
└── plan.md
```

---

## 단계별 설계

### ✅ 1단계 - dataset/

**`dataset/mnist_dataset.py`**
- `MNISTDataset(Dataset)` 클래스
  - `__init__(self, data_dir, train, transform, visualization=False)`
  - `__getitem__(self, idx)`: `visualization=True`이면 해당 샘플을 matplotlib으로 표시
  - `__len__`
- `__main__` 블록: `visualization=True`로 첫 샘플 표시 확인

**`dataset/mnist_dataloader.py`**
- `build_dataloader(data_dir="./data", batch_size=64, visualization=False)` → `(train_loader, test_loader)`
  - `MNISTDataset`에 `visualization` 플래그 전달
  - 기본 transform: ToTensor + Normalize(mean=0.1307, std=0.3081)
- `__main__` 블록: 첫 배치 shape 출력 확인

기존 파일 처리:
- `mnist_dataset.py` (루트) → `dataset/mnist_dataset.py`로 통합 후 제거
- `mnist_visualize.py` (루트) → visualization 기능을 Dataset으로 흡수 후 제거

---

### ✅ 2단계 - model/

**`model/transformer.py`**
- `MNISTClassifier(nn.Module)` 클래스
  - `__init__(embed_dim=128, depth=6, num_heads=8, patch_size=4, num_classes=10)`
  - `forward(x, t)` → `(B, 10)` 로짓 출력
- `get_model()` → 기본 하이퍼파라미터로 `MNISTClassifier` 인스턴스 반환
- `__main__` 블록: 더미 입력 `x=(1,1,28,28)`, `t=(1,)` 통과 후 출력 shape `(1, 10)` 확인

**아키텍처 상세**

```
입력: x_t (B, 1, 28, 28),  t (B,)
│
├─ [Patchify]
│    4×4 패치 분할 → (B, 49, 16)
│    Linear(16 → 128) → (B, 49, 128)
│
├─ [2D RoPE]
│    각 패치의 (row, col) 위치를 기반으로 상대 위치 인코딩
│    → Self-Attention의 Q, K에 적용 (학습 파라미터 없음)
│    → 나중에 해상도 변경 시 재학습 없이 확장 가능
│
├─ [시간 임베딩]
│    t → Sinusoidal embedding(128) → MLP(128→256→128)
│
├─ [Transformer Block × 6]
│    각 블록 구조:
│    ├─ Self-Attention (heads=8, head_dim=16) with 2D RoPE
│    ├─ adaLN: γ(t)·LayerNorm(x) + β(t)   ← FiLM, 블록별 독립 γ/β
│    └─ FFN: Linear(128→512→128) + GELU
│
├─ [분류 헤드]
│    LayerNorm → 평균 풀링(시퀀스 축) → Linear(128→10)
│
└─ 출력: (B, 10) 로짓  ← CrossEntropyLoss와 짝
```

**설계 근거**
- `embed_dim=128, depth=6, num_heads=8, head_dim=16`: MNIST 스케일에 적합한 경량 구성
- 2D RoPE: 절대 위치 임베딩 대신 사용, 입력 해상도 확장 시 재학습 불필요
- adaLN(FiLM): 각 블록이 `t`를 보고 독립적으로 feature scale/shift 조절
  → 모델이 입력의 노이즈 강도(어려움)를 레이어별로 판단 가능
- 시간 임베딩 MLP: sinusoidal → 선형 변환으로 충분한 표현력 확보

---

### ⬜ 3단계 - train/ (TBD)

추후 논의

---

### ⬜ 4단계 - eval/ (TBD)

추후 논의

---

## 구현 체크리스트

### 1단계 - dataset/
- [x] `dataset/mnist_dataset.py` — `MNISTDataset` 클래스 구현
- [x] `dataset/mnist_dataloader.py` — `build_dataloader()` 구현
- [x] 루트 `mnist_dataset.py`, `mnist_visualize.py` 제거
- [x] `python dataset/mnist_dataset.py` 실행 확인 (이미지 표시, shape `[1, 28, 28]`)
- [x] `python dataset/mnist_dataloader.py` 실행 확인 (배치 shape `(64, 1, 28, 28)`, 배치 수 938/10)

### 2단계 - model/
- [x] `model/transformer.py` — `MNISTClassifier` 구현 (Patchify + 2D RoPE + Transformer×6 + adaLN)
- [x] `python model/transformer.py` 실행 확인 (출력 shape `(1, 10)`, 파라미터 1,653,258)

### 3단계 - train/
- [ ] TBD

### 4단계 - eval/
- [ ] TBD
