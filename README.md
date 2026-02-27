# MNIST Classification with Flow Matching

시간 `t` 조건부 Transformer 기반 MNIST 분류기.
Flow Matching 환경에서 노이즈가 섞인 이미지 `x_t`와 노이즈 강도 `t`를 입력받아 클래스를 예측한다.

---

## 모델 구조

```
x_t (B, 1, 28, 28)  +  t (B,)
        │                  │
   Patchify (4×4)     Sinusoidal Embedding
   Linear(16→128)     MLP(128→256→128)
        │                  │
        └──────┬────────────┘
               ↓
     Transformer Block × 6
     ┌─ Self-Attention (heads=8) + 2D RoPE
     └─ FFN (128→512→128) + adaLN(γ(t), β(t))
               ↓
     평균 풀링 → Linear(128→10)
               ↓
         로짓 (B, 10)
```

| 항목 | 값 |
|------|-----|
| 패치 크기 | 4×4 → 49개 패치 |
| embed_dim | 128 |
| Transformer 블록 | 6 |
| Attention 헤드 | 8 (head_dim=16) |
| 위치 인코딩 | 2D RoPE (해상도 확장 가능) |
| t 주입 방식 | adaLN — 블록별 독립 γ(t), β(t) |
| 총 파라미터 | 1,653,258 |

---

## 프로젝트 구조

```
mnist_classifiation_fm/
├── dataset/
│   ├── mnist_dataset.py      # MNISTDataset 클래스
│   └── mnist_dataloader.py   # build_dataloader() 함수
├── model/
│   └── transformer.py        # MNISTClassifier (Transformer + 2D RoPE + adaLN)
├── train/                    # 학습 루프 (예정)
├── eval/                     # 평가 (예정)
└── data/                     # MNIST 원본 데이터 (자동 다운로드)
```

---

## 설치

```bash
conda create -n mnist_fm python=3.9
conda activate mnist_fm
pip install -r requirements.txt
```

---

## 실행

### 데이터셋 확인
```bash
python dataset/mnist_dataset.py    # 샘플 이미지 시각화
python dataset/mnist_dataloader.py # 배치 shape 확인
```

### 모델 동작 확인
```bash
python model/transformer.py
# 입력 x shape : [1, 1, 28, 28]
# 입력 t       : 0.5
# 출력 logits  : [1, 10]
# 총 파라미터  : 1,653,258
```

---

## 진행 상황

- [x] 1단계 — dataset (MNISTDataset, DataLoader)
- [x] 2단계 — model (Transformer + 2D RoPE + adaLN)
- [ ] 3단계 — train
- [ ] 4단계 — eval
