import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 2D RoPE
# ─────────────────────────────────────────────────────────────────

def compute_rope_inv_freq(dim: int, base: int = 10000, device=None) -> torch.Tensor:
    """RoPE 역주파수 계산.

    Args:
        dim: 적용할 차원 수 (짝수여야 함)
    Returns:
        inv_freq: (dim//2,)
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    return inv_freq


def apply_rope_1d(
    x: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """1D RoPE 적용.

    Args:
        x        : (B, heads, N, dim)
        positions: (N,)  — 정수 위치 인덱스
        inv_freq : (dim//2,)
    Returns:
        (B, heads, N, dim)
    """
    freqs = torch.outer(positions.float(), inv_freq)   # (N, dim//2)
    cos = freqs.cos()[None, None]                      # (1, 1, N, dim//2)
    sin = freqs.sin()[None, None]

    x1 = x[..., 0::2]   # (B, heads, N, dim//2)
    x2 = x[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)                         # (B, heads, N, dim)


def apply_2d_rope(
    x: torch.Tensor,
    row_pos: torch.Tensor,
    col_pos: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """2D RoPE 적용 — head_dim을 절반으로 나눠 row/col 각각 1D RoPE 적용.

    Args:
        x       : (B, heads, N, head_dim)
        row_pos : (N,)
        col_pos : (N,)
        inv_freq: (head_dim//4,)  ← head_dim 절반에 대한 1D RoPE 역주파수
    """
    half = x.shape[-1] // 2
    x_r = apply_rope_1d(x[..., :half], row_pos, inv_freq)
    x_c = apply_rope_1d(x[..., half:], col_pos, inv_freq)
    return torch.cat([x_r, x_c], dim=-1)


# ─────────────────────────────────────────────────────────────────
# 시간 임베딩
# ─────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """t ∈ [0,1] → sinusoidal 고정 임베딩 (학습 파라미터 없음)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, dim)"""
        half = self.dim // 2
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half, device=t.device).float() / half)
        )
        angles = t.float()[:, None] * inv_freq[None]   # (B, half)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    """sinusoidal embedding → MLP(dim → dim*2 → dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.sin_emb = SinusoidalEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin_emb(t))   # (B, dim)


# ─────────────────────────────────────────────────────────────────
# Transformer 블록 구성 요소
# ─────────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        row_pos: torch.Tensor,
        col_pos: torch.Tensor,
        inv_freq: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        q = apply_2d_rope(q, row_pos, col_pos, inv_freq)
        k = apply_2d_rope(k, row_pos, col_pos, inv_freq)

        scale = self.head_dim ** -0.5
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Self-Attention + FFN, 각각 adaLN(FiLM)으로 t 조건화."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads)
        self.ffn = FFN(embed_dim)

        # t_emb → [γ1, β1, γ2, β2], 초기값 0 (γ에 +1 더해 identity 초기화)
        self.adaLN = nn.Linear(embed_dim, 4 * embed_dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        row_pos: torch.Tensor,
        col_pos: torch.Tensor,
        inv_freq: torch.Tensor,
    ) -> torch.Tensor:
        γ1, β1, γ2, β2 = self.adaLN(t_emb).chunk(4, dim=-1)  # 각 (B, D)
        γ1 = (γ1 + 1).unsqueeze(1)   # (B, 1, D) — +1: identity 초기화
        β1 = β1.unsqueeze(1)
        γ2 = (γ2 + 1).unsqueeze(1)
        β2 = β2.unsqueeze(1)

        x = x + self.attn(γ1 * self.norm1(x) + β1, row_pos, col_pos, inv_freq)
        x = x + self.ffn(γ2 * self.norm2(x) + β2)
        return x


# ─────────────────────────────────────────────────────────────────
# 메인 모델
# ─────────────────────────────────────────────────────────────────

class MNISTClassifier(nn.Module):
    """시간 t 조건부 MNIST 분류기.

    x_t (노이즈 이미지)와 t (노이즈 강도)를 입력받아 클래스 로짓 출력.
    2D RoPE + adaLN(FiLM) 기반 Transformer 구조.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        patch_size: int = 4,
        num_classes: int = 10,
        img_size: int = 28,
        in_channels: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.head_dim = embed_dim // num_heads   # 16

        grid_size = img_size // patch_size       # 7
        patch_dim = patch_size * patch_size * in_channels   # 16

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.time_embed = TimeEmbedding(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # RoPE 역주파수: head_dim 절반(8)에 대한 1D RoPE → (4,)
        inv_freq = compute_rope_inv_freq(self.head_dim // 2)
        self.register_buffer("inv_freq", inv_freq)

        # 패치 (row, col) 위치 인덱스 — 고정값
        rows = torch.arange(grid_size).repeat_interleave(grid_size)  # (49,)
        cols = torch.arange(grid_size).repeat(grid_size)              # (49,)
        self.register_buffer("row_pos", rows)
        self.register_buffer("col_pos", cols)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, patch_dim)"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)   # (B, H/p, W/p, C, p, p)
        x = x.flatten(1, 2)               # (B, N, C, p, p)
        return x.flatten(2)               # (B, N, patch_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 28, 28) — 노이즈 이미지
            t: (B,)           — 노이즈 강도 t ∈ [0, 1]
        Returns:
            logits: (B, 10)
        """
        x = self.patch_embed(self.patchify(x))   # (B, 49, 128)
        t_emb = self.time_embed(t)               # (B, 128)

        for block in self.blocks:
            x = block(x, t_emb, self.row_pos, self.col_pos, self.inv_freq)

        x = self.norm(x).mean(dim=1)   # (B, 128) — 시퀀스 평균 풀링
        return self.head(x)            # (B, 10)


def get_model(**kwargs) -> MNISTClassifier:
    """기본 하이퍼파라미터로 MNISTClassifier 반환."""
    return MNISTClassifier(**kwargs)


if __name__ == "__main__":
    model = get_model()
    total_params = sum(p.numel() for p in model.parameters())

    x = torch.randn(1, 1, 28, 28)
    t = torch.tensor([0.5])

    logits = model(x, t)

    print(f"입력 x shape : {list(x.shape)}")
    print(f"입력 t       : {t.item()}")
    print(f"출력 logits  : {list(logits.shape)}")
    print(f"총 파라미터  : {total_params:,}")
