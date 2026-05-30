import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# --- 动态路径挂载 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
VMAMBA_ROOT = os.path.join(PROJECT_ROOT, 'VMamba')

paths = [VMAMBA_ROOT, os.path.join(VMAMBA_ROOT, 'classification')]
for p in paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# --- 导入 VMamba 组件（带降级保护）---
try:
    from models.vmamba import VSSBlock
except Exception as e:
    print(f"⚠️ [WARNING] 无法加载 VMamba 算子，降级为占位符: {e}")
    class VSSBlock(nn.Module):
        def __init__(self, hidden_dim=32, **kwargs):
            super().__init__()
            self.proj = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
            self.act = nn.SiLU()
        def forward(self, x):
            return x + self.act(self.proj(x))


# --- 3. VMamba 包装器：确保尺寸不变 ---
class VSSBlockWrapper(nn.Module):
    """
    确保 VSSBlock 输入输出尺寸严格不变，并处理 [B,C,H,W] ↔ [B,H,W,C] 格式转换。
    """
    def __init__(self, hidden_dim=32, drop_path=0.1):
        super().__init__()
        self.vss = VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            channel_first=False,   # 内部期望 [B, H, W, C]
            downsample=None        # 禁止下采样
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()   # [B, H, W, C]
        x = self.vss(x)
        if x.shape[1] != H or x.shape[2] != W:    # 尺寸意外变化时强制插值
            x = x.permute(0, 3, 1, 2)             # [B, C, H', W']
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x.permute(0, 2, 3, 1)             # [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()    # [B, C, H, W]
        return x


# --- 4. 统一双域自编码器（单域/双域可切换）---
class DualDomainAE(nn.Module):
    """
    统一自编码器，通过 use_freq_branch 控制是否启用频域分支。

    参数:
        latent_dim (int): 潜在空间维度，默认 32。
        use_freq_branch (bool): 是否启用频域分支。
            - True: 双域模式（空间+频域），用于第二阶段训练和检测。
            - False: 纯空间域模式，等价于原 SpatialOnlyAE，用于第一阶段预训练。
        freq_input_channels (int): 频域分支输入通道数。
            - 3: 直接使用 RGB 三通道的频谱（默认）。
            - 1: 内部转为灰度图后计算单通道频谱（更高效）。
    """
    def __init__(self, latent_dim=32, use_freq_branch=True, freq_input_channels=3):
        super().__init__()
        self.use_freq_branch = use_freq_branch
        self.freq_input_channels = freq_input_channels

        # ---------- 空间分支（共享） ----------
        self.spatial_stem = nn.Conv2d(3, 32, 3, 1, 1)
        self.vss1 = VSSBlockWrapper(32, drop_path=0.1)
        self.vss2 = VSSBlockWrapper(32, drop_path=0.1)
        self.spatial_to_latent = nn.Conv2d(32, latent_dim, 1)

        # ---------- 频域分支（可选） ----------
        if use_freq_branch:
            self.freq_enc = nn.Sequential(
                nn.Conv2d(freq_input_channels, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, latent_dim, 3, 1, 1)
            )
            dec_in_channels = latent_dim * 2
        else:
            dec_in_channels = latent_dim

        # ---------- 解码器 ----------
        self.dec = nn.Sequential(
            nn.Conv2d(dec_in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 空间分支
        s = self.spatial_stem(x)
        s = self.vss1(s)
        s = self.vss2(s)
        s_feat = self.spatial_to_latent(s)

        if self.use_freq_branch:
            # 频域分支：可根据 freq_input_channels 选择是否转灰度
            if self.freq_input_channels == 1:
                # 转为单通道灰度图
                gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
                freq = torch.fft.fft2(gray, dim=(-2, -1), norm="ortho")
            else:
                freq = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
            amp = torch.log(torch.abs(freq) + 1e-8)
            f_feat = self.freq_enc(amp)

            fused = torch.cat([s_feat, f_feat], dim=1)
        else:
            fused = s_feat
            f_feat = None

        rec = self.dec(fused)
        # 确保输出尺寸与输入一致（正常情况下 CIFAR-10 已是 32x32）
        if rec.shape[-2:] != x.shape[-2:]:
            rec = F.interpolate(rec, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return rec, s_feat, f_feat