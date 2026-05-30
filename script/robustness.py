import sys
import warnings
from pathlib import Path
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gc
import random

matplotlib.use("Agg")

# ========== 固定全局随机种子（确保可复现） ==========
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# ================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.encoder import DualDomainAE
    from src.data_loader import get_loaders
    import src.detector as detector
    from src.baselines.mse import MSE
    from src.baselines.ocsvm import OCSVM
    from src.attack_generator import generate_adv_dataset
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")


def setup_chinese_font():
    font_path = PROJECT_ROOT / "simhei.ttf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        prop = fm.FontProperties(fname=str(font_path))
        plt.rcParams['font.sans-serif'] = [prop.get_name()]
    else:
        print("⚠️ 未找到 simhei.ttf，使用系统默认字体")
    plt.rcParams['axes.unicode_minus'] = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
ADV_DIR = PROJECT_ROOT / "saved_adv_samples"

EPS_VALUES = [4/255, 8/255, 12/255, 16/255]
REPRESENTATIVE_EPS = 8/255

# 目标样本数量（初始设为较大值，后续动态调整为实际最小值）
TARGET_SAMPLES = 200


def get_adv_file_path(eps):
    eps_int = int(round(eps * 255))
    return ADV_DIR / f"Adv_DI_MI_TEST_E{eps_int}_N200.pt"


def generate_missing_files():
    """生成缺失的对抗样本文件（若失败则跳过）"""
    for eps in EPS_VALUES:
        file_path = get_adv_file_path(eps)
        if file_path.exists():
            print(f"  ✅ 已存在: {file_path.name}")
        else:
            print(f"  ⚠️ 缺失: {file_path.name}，开始生成...")
            try:
                path = generate_adv_dataset(
                    need_count=TARGET_SAMPLES,
                    use_train_set=False,
                    force=True,
                    batch_size=16,
                    eps=eps,
                    psnr_threshold=25.0
                )
                if path is None:
                    print(f"    ❌ 生成失败：未能收集到合格样本，跳过此扰动强度。")
                else:
                    print(f"    ✅ 生成完毕: {Path(path).name}")
            except Exception as e:
                print(f"    ❌ 生成异常: {e}")
            finally:
                torch.cuda.empty_cache()
                gc.collect()
    print("✅ 对抗样本文件检查完成。\n")


def get_actual_sample_count(file_path):
    """返回文件中实际保存的对抗样本数量"""
    if not file_path.exists():
        return 0
    data = torch.load(file_path, map_location='cpu')
    return data['adv_images'].size(0)


def evaluate_metrics(model, images, m_name, mse_bl, ocsvm_bl, is_adv=True):
    if len(images) == 0:
        return 0.0
    all_preds = []
    model.eval()
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            if m_name == "DUAD":
                p = detector.is_adversarial_batch(model, batch, device=DEVICE)
            elif m_name == "MSE":
                out = model(batch)
                rec = out[0] if isinstance(out, (list, tuple)) else out
                mse_s = torch.mean((batch - rec) ** 2, dim=(1, 2, 3))
                p = (mse_s > mse_bl.threshold)
                del out, rec
            else:  # OC-SVM
                lat = ocsvm_bl.get_latent(model, batch)
                lat_p = ocsvm_bl.pca.transform(ocsvm_bl.scaler.transform(lat))
                p_np = ocsvm_bl.model.predict(lat_p)
                p = torch.from_numpy(p_np == -1).to(DEVICE)
                del lat, lat_p, p_np
        all_preds.append(p.cpu())
        del batch, p
        torch.cuda.empty_cache()
    rate = torch.cat(all_preds).float().mean().item() * 100
    del all_preds
    torch.cuda.empty_cache()
    return rate if is_adv else (100.0 - rate)


def load_adv_samples(file_path, count):
    data = torch.load(file_path, map_location='cpu')
    return data['adv_images'][:count].to(DEVICE)


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


def run_robustness_experiment():
    setup_chinese_font()
    save_dir = PROJECT_ROOT / "result" / "robustness"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 步骤1：生成/检查对抗样本文件 ----------
    generate_missing_files()

    # ---------- 步骤2：确定统一使用的样本数量 ----------
    print("📊 检查各扰动强度实际可用样本数：")
    available_counts = {}
    for eps in EPS_VALUES:
        cnt = get_actual_sample_count(get_adv_file_path(eps))
        available_counts[eps] = cnt
        print(f"  eps={eps*255:.0f}/255: {cnt} 张")
    valid_eps = [eps for eps in EPS_VALUES if available_counts[eps] > 0]
    if not valid_eps:
        print("❌ 没有任何可用的对抗样本，实验终止。")
        return
    unified_count = min(available_counts[eps] for eps in valid_eps)
    print(f"🔔 采用统一样本数量: {unified_count} 张（取各扰动强度最小值）\n")

    # ---------- 步骤3：加载自编码器模型 ----------
    model = DualDomainAE(latent_dim=8).to(DEVICE)
    model_path = PROJECT_ROOT / "checkpoint" / "vmamba" / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # ---------- 步骤4：初始化基线检测器 ----------
    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)
    print("🧪 正在初始化基准统计参数...")
    detector.init_threshold(model, train_loader, device=DEVICE)
    torch.cuda.empty_cache()

    mse_baseline = MSE()
    mse_baseline.fit(model, train_loader)
    torch.cuda.empty_cache()

    ocsvm_baseline = OCSVM()
    ocsvm_baseline.fit(model, train_loader)
    torch.cuda.empty_cache()

    methods = ["DUAD", "MSE", "OC-SVM"]
    style_configs = [
        {'color': '#C44E52', 'marker': 'o', 'ls': '-', 'label': 'DUAD'},
        {'color': '#4C72B0', 'marker': 'D', 'ls': '--', 'label': 'MSE'},
        {'color': '#55A868', 'marker': '^', 'ls': ':', 'label': 'OC-SVM'}
    ]

    # ---------- 步骤5：计算 TNR（使用统一数量）----------
    tnr_imgs = collect_n_images(test_loader, unified_count, DEVICE)
    tnr_results = [evaluate_metrics(model, tnr_imgs, m, mse_baseline, ocsvm_baseline, is_adv=False) for m in methods]
    print("\n📊 正常样本放行率 (TNR):")
    for m, v in zip(methods, tnr_results):
        print(f"  {m}: {v:.2f}%")
    del tnr_imgs
    torch.cuda.empty_cache()

    # ---------- 步骤6：评估各 eps 下的 TPR ----------
    tpr_data = {eps: [] for eps in valid_eps}
    for eps in valid_eps:
        file_path = get_adv_file_path(eps)
        adv_imgs = load_adv_samples(file_path, unified_count)
        print(f"\n🔍 评估 eps={eps*255:.0f}/255 (使用 {unified_count} 张样本)...")
        for m in methods:
            tpr = evaluate_metrics(model, adv_imgs, m, mse_baseline, ocsvm_baseline, is_adv=True)
            tpr_data[eps].append(tpr)
            print(f"  {m}: TPR = {tpr:.2f}%")
        del adv_imgs
        torch.cuda.empty_cache()

    # ---------- 步骤7：绘图 ----------
    # 曲线图
    plt.figure(figsize=(10, 6))
    eps_ticks = [e * 255 for e in valid_eps]
    for i, m in enumerate(methods):
        tpr_curve = [tpr_data[e][i] for e in valid_eps]
        conf = style_configs[i]
        plt.plot(eps_ticks, tpr_curve, label=conf['label'], color=conf['color'],
                 marker=conf['marker'], ls=conf['ls'], lw=2.5, markersize=8,
                 markeredgecolor='white', markeredgewidth=1.5)
    plt.title(f"对抗样本检测率随扰动强度变化折线 (样本数={unified_count})", fontsize=14)
    plt.xlabel("扰动强度 $\epsilon$")
    plt.ylabel("对抗检测率 (TPR %)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "tpr_vs_eps.png", dpi=300)
    plt.close()
    print(f"✅ 曲线图已保存: {save_dir / 'tpr_vs_eps.png'}")

    # 代表性柱状图 (eps=8/255)
    if REPRESENTATIVE_EPS in valid_eps:
        tprs_rep = tpr_data[REPRESENTATIVE_EPS]
        plt.figure(figsize=(10, 5))
        x_axis = np.arange(len(methods))
        plt.bar(x_axis - 0.2, tnr_results, 0.4, label='正常放行率 (TNR)', color='#55A868', alpha=0.7)
        plt.bar(x_axis + 0.2, tprs_rep, 0.4, label='对抗检测率 (TPR)', color='#C44E52', alpha=0.7)
        plt.xticks(x_axis, methods)
        plt.ylabel("百分比 (%)")
        plt.ylim(0, 115)
        plt.legend()
        plt.title(f"标准扰动强度 $\epsilon$={REPRESENTATIVE_EPS*255:.0f}/255 下各方法性能对比 (样本数={unified_count})", fontsize=14)
        for i in range(len(methods)):
            plt.text(i - 0.2, tnr_results[i] + 1, f'{tnr_results[i]:.1f}%', ha='center', fontweight='bold')
            plt.text(i + 0.2, tprs_rep[i] + 1, f'{tprs_rep[i]:.1f}%', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / "bar_eps_8.png", dpi=300)
        plt.close()
        print(f"✅ 柱状图已保存: {save_dir / 'bar_eps_8.png'}")

    print(f"\n🎉 鲁棒性实验完成！结果存放在: {save_dir}")


if __name__ == "__main__":
    run_robustness_experiment()