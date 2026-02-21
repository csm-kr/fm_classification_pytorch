# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MNIST 데이터 다운로드 및 시각화 파이프라인. 데이터 준비와 모델학습까지 모든것을 준비한다. 

## Dependencies

환경 관리: conda (VSCode Python 확장 기준)

## Dataset 

MNIST 데이터는 `./data/MNIST/raw/` 에 캐시된다. 이미 존재하면 재다운로드하지 않는다.

## Architecture

폴더별로 역할이 분리된다:

```
mnist_classifiation_fm/
├── dataset/
│   ├── mnist_dataset.py      # MNISTDataset 클래스 (visualization 포함)
│   └── mnist_dataloader.py   # build_dataloader() 함수
├── model/                    # TBD
├── train/                    # TBD
├── eval/                     # TBD
└── data/                     # MNIST 원본 데이터 (자동 다운로드)
```

## Conventions

각 `.py` 모듈 파일에는 `if __name__ == "__main__":` 블록을 두어, 해당 파일의 핵심 함수를 직접 호출하는 최소한의 동작 확인 코드를 포함한다.

```python
if __name__ == "__main__":
    train_dataset, test_dataset = get_mnist_datasets()
```
이 블록은 별도의 테스트 프레임워크 없이 `python <파일명>.py`로 즉시 동작을 검증하기 위한 용도다.

## 언어 규칙

- 모든 결과값과 설명은 반드시 한글로 작성한다.
- 코드 주석, 커밋 메시지, 사용자 응답 등 모든 텍스트 출력은 한글을 기본으로 한다.

## Git 규칙

### 브랜치 전략

```
master          # 안정적인 릴리즈 버전
└── develop     # 개발 통합 브랜치
    └── feature/피쳐이름  # 기능 개발 브랜치
```

- `feature/*` 브랜치는 `develop`에 병합한다.
- `develop`이 안정화되면 `master`에 병합한다.
- 브랜치 이름은 소문자와 하이픈을 사용한다. (예: `feature/mnist-dataset`, `feature/dataloader`)

### 커밋 단위

- 하나의 기능 단위로 커밋한다. (여러 파일이라도 하나의 기능이면 한 커밋)
- 커밋 메시지는 한글로 작성한다.
- 형식: `타입: 설명`

| 타입 | 용도 |
|------|------|
| `feat` | 새 기능 추가 |
| `fix` | 버그 수정 |
| `refactor` | 기능 변경 없는 코드 개선 |
| `docs` | 문서 수정 |
| `chore` | 빌드·설정 등 기타 변경 |

예시:
```
feat: MNISTDataset 클래스 구현
fix: 데이터 경로 오류 수정
refactor: build_dataloader 함수 분리
```

## Scope

- 포함: 데이터 다운로드, 랜덤 샘플 시각화, 클래스별(0~9) 샘플 시각화
- 제외: 모델 학습/평가, 실험 추적 도구, argparse 확장
