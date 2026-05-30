import torch
import torch.nn.functional as F
import numpy as np

# -------------------- 全局物理阈值 --------------------
SPATIAL_MSE_THRESH = 0.0
FREQ_MSE_THRESH = 0.0
PHASE_DIFF_THRESH = 0.30          # 固定经验阈值，平衡 TPR 与 FPR

# -------------------- 灵敏度系数 --------------------
_SENSITIVITY = 1.5

# -------------------- 辅助工具 --------------------

def get_amp_spectrum(img):
    """提取对数幅度谱"""
    fft_res = torch.fft.fft2(img, dim=(-2, -1), norm="ortho")
    return torch.log1p(torch.abs(fft_res))


def get_phase_spectrum(img):
    """提取相位谱（中心化），返回 [B,1,H,W] 张量"""
    gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    fft = torch.fft.fft2(gray, dim=(-2, -1), norm="ortho")
    fft_shifted = torch.fft.fftshift(fft)
    return torch.angle(fft_shifted)


def compute_phase_diff(phase1, phase2):
    """计算两个相位谱的平均绝对角度差（弧度）"""
    diff = torch.angle(torch.exp(1j * phase1) / torch.exp(1j * phase2))
    return torch.abs(diff).mean(dim=(1, 2, 3))


def apply_bit_reduction(x, bits=4):
    """位深度归约（保留，未使用）"""
    levels = 2 ** bits
    return torch.round(x * (levels - 1)) / (levels - 1)


# -------------------- 阈值校准逻辑 --------------------

def init_threshold(model, loader, device):
    """
    仅校准空间 MSE 和频域幅度 MSE 阈值（使用 IQR）。
    相位差异阈值已固定，无需校准。
    """
    global SPATIAL_MSE_THRESH, FREQ_MSE_THRESH
    metrics = {'s': [], 'f': []}

    model.eval()
    print(f"⏳ [Detector] 正在提取数据集分布特征 (灵敏度系数: {_SENSITIVITY})...")

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            rec_orig, _, _ = model(x)

            mse_s = torch.mean((rec_orig - x) ** 2, dim=(1, 2, 3))
            mse_f = torch.mean((get_amp_spectrum(rec_orig) - get_amp_spectrum(x)) ** 2, dim=(1, 2, 3))

            metrics['s'].append(mse_s.cpu())
            metrics['f'].append(mse_f.cpu())

    def get_robust_threshold(data_list):
        arr = torch.cat(data_list).numpy()
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return q3 + _SENSITIVITY * iqr

    SPATIAL_MSE_THRESH = get_robust_threshold(metrics['s'])
    FREQ_MSE_THRESH = get_robust_threshold(metrics['f'])

    print(f"✅ 校准完成！")
    print(f"  [空间 MSE 阈值]: {SPATIAL_MSE_THRESH:.6f}")
    print(f"  [频域幅度 MSE 阈值]: {FREQ_MSE_THRESH:.6f}")
    print(f"  [相位差异阈值]: {PHASE_DIFF_THRESH:.4f} rad (固定值，联合条件)")


# -------------------- 核心检测接口 --------------------

def is_adversarial_batch(model, batch, device):
    """
    联合判定逻辑：
    1. 空间 MSE 超过阈值 → 对抗
    2. 频域幅度 MSE 超过阈值 → 对抗
    3. 相位差超过阈值 **且** 空间 MSE 超过阈值的 70% → 对抗
    """
    model.eval()
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    batch = batch.to(device)

    with torch.no_grad():
        rec_orig, _, _ = model(batch)

        s_mse = torch.mean((rec_orig - batch) ** 2, dim=(1, 2, 3))
        f_mse = torch.mean((get_amp_spectrum(rec_orig) - get_amp_spectrum(batch)) ** 2, dim=(1, 2, 3))

        phase_x = get_phase_spectrum(batch)
        phase_rec = get_phase_spectrum(rec_orig)
        phase_diff = compute_phase_diff(phase_x, phase_rec)

    # 基础判定
    base_hit = (s_mse > SPATIAL_MSE_THRESH) | (f_mse > FREQ_MSE_THRESH)

    # 联合条件：相位差高 且 空间MSE较高
    phase_hit = (phase_diff > PHASE_DIFF_THRESH) & (s_mse > SPATIAL_MSE_THRESH * 0.6)

    is_adv = base_hit | phase_hit
    return is_adv


def get_raw_mse(model, x, device):
    """返回空间 MSE 和频域幅度 MSE"""
    model.eval()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        rec, _, _ = model(x)
        s_mse = torch.mean((rec - x) ** 2, dim=(1, 2, 3))
        f_mse = torch.mean((get_amp_spectrum(rec) - get_amp_spectrum(x)) ** 2, dim=(1, 2, 3))
    return s_mse, f_mse


def get_phase_diff_score(model, x, device):
    """返回基于重建的相位差异分数"""
    model.eval()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        rec, _, _ = model(x)
        phase_x = get_phase_spectrum(x)
        phase_rec = get_phase_spectrum(rec)
        return compute_phase_diff(phase_x, phase_rec)


def apply_dual_domain_defense(model, x, device):
    """净化接口：返回重建图像"""
    model.eval()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        rec, _, _ = model(x)
    return rec