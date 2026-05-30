# script/compare.py

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.encoder import DualDomainAE
import src.detector as detector
from src.data_loader import get_loaders

# -------------------- 中文字体设置 --------------------
def setup_chinese_font():
    font_path = PROJECT_ROOT / "simhei.ttf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        prop = fm.FontProperties(fname=str(font_path))
        font_name = prop.get_name()
        plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 已加载中文字体: {font_path}")
    else:
        print("⚠️ 未找到 simhei.ttf，图表可能乱码")

# -------------------- 配置 --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = PROJECT_ROOT / "checkpoint" / "vmamba" / "best_model.pth"
ADV_DATA_PATH = PROJECT_ROOT / "saved_adv_samples" / "Adv_DI_MI_TEST_E8_N1000.pt"
TEST_SAMPLES = 200
BATCH_SIZE = 64
SAVE_PATH = PROJECT_ROOT / "result" / "compare" / "optimization_comparison.png"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------------------- 工具函数 --------------------
def collect_n_images(loader, n, device):
    imgs = []
    cnt = 0
    for x, _ in loader:
        if cnt >= n:
            break
        take = min(n - cnt, x.size(0))
        imgs.append(x[:take].to(device))
        cnt += take
    return torch.cat(imgs, dim=0)

def evaluate_mode(model, images, mode, device, sp_th, fr_th, ph_th):
    all_preds = []
    model.eval()
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE].to(device)
        with torch.no_grad():
            rec, _, _ = model(batch)
            s_mse = torch.mean((rec - batch) ** 2, dim=(1,2,3))
            f_mse = torch.mean((detector.get_amp_spectrum(rec) - detector.get_amp_spectrum(batch))**2, dim=(1,2,3))
            phase_diff = detector.get_phase_diff_score(model, batch, DEVICE)
            if mode == "baseline":
                p = s_mse > sp_th
            else:  # DUAD
                base = (s_mse > sp_th) | (f_mse > fr_th)
                phase_hit = (phase_diff > ph_th) & (s_mse > sp_th * 0.6)
                p = base | phase_hit
        all_preds.append(p.cpu())
    rate = torch.cat(all_preds).float().mean().item() * 100
    return rate

# -------------------- 主流程 --------------------
def main():
    setup_chinese_font()

    # 加载模型与阈值
    model = DualDomainAE(latent_dim=8).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)
    detector.init_threshold(model, train_loader, DEVICE)

    sp_th = detector.SPATIAL_MSE_THRESH
    fr_th = detector.FREQ_MSE_THRESH
    ph_th = detector.PHASE_DIFF_THRESH

    # 加载样本
    data = torch.load(ADV_DATA_PATH, map_location='cpu')
    norm_imgs = data['norm_images'][:TEST_SAMPLES].to(DEVICE)
    adv_imgs = data['adv_images'][:TEST_SAMPLES].to(DEVICE)
    tnr_imgs = collect_n_images(test_loader, TEST_SAMPLES, DEVICE)

    # 评估两种模式
    modes = [("baseline", "原生自编码器"), ("duad", "DUAD")]
    tnr_list, tpr_list, labels = [], [], []

    for mode, label in modes:
        tnr = 100.0 - evaluate_mode(model, tnr_imgs, mode, DEVICE, sp_th, fr_th, ph_th)
        tpr = evaluate_mode(model, adv_imgs, mode, DEVICE, sp_th, fr_th, ph_th)
        tnr_list.append(tnr)
        tpr_list.append(tpr)
        labels.append(label)

    # 绘图（仅TNR与TPR）
    x = np.arange(len(labels))
    width = 0.3
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, tnr_list, width, label='正常放行率 (TNR)', color='#55A868', alpha=0.85)
    bars2 = ax.bar(x + width/2, tpr_list, width, label='对抗检测率 (TPR)', color='#C44E52', alpha=0.85)

    # 数值标签
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('百分比 (%)')
    ax.set_title('DUAD 与原生自编码器检测性能对比 (ε=8/255)', fontsize=13)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"✅ 对比图已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    main()