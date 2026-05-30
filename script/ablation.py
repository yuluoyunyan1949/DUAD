import sys
import torch
import numpy as np
import matplotlib
import os
import warnings
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ==========================
# 1. 路径与环境配置
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.attack_generator import generate_adv_dataset  # 仅保留此函数
    from src.encoder import DualDomainAE
    from src.data_loader import get_loaders
    import src.detector as detector
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")


# ==========================
# 2. 辅助功能
# ==========================
def setup_chinese_font():
    font_file = PROJECT_ROOT / "simhei.ttf"
    if font_file.exists():
        fe = fm.FontEntry(fname=str(font_file), name='SimHei_Local')
        fm.fontManager.ttflist.insert(0, fe)
        plt.rcParams['font.sans-serif'] = ['SimHei_Local']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def load_local_classifiers(device):
    from torch.hub import load as hub_load
    repo = "chenyaofo/pytorch-cifar-models"
    model_map = {'resnet20': 'resnet20', 'vgg16_bn': 'vgg16_bn', 'mobileNetV2': 'mobilenetv2_x1_0'}
    sources = []
    print("⏳ 正在检索分类器作为攻击源...")
    for name, hub_func in model_map.items():
        try:
            m = hub_load(repo, f"cifar10_{hub_func}", pretrained=False)
            # 修正为实际存在的权重路径
            ckpt = PROJECT_ROOT / "checkpoint" / name / "mfiae_model.pt"
            if ckpt.exists():
                sd = torch.load(ckpt, map_location=device, weights_only=True)
                m.load_state_dict(sd.get('state_dict', sd.get('model', sd)))
                sources.append((name, m.to(device).eval()))
        except:
            continue
    if not sources:
        m = hub_load(repo, "cifar10_resnet20", pretrained=True)
        sources.append(('resnet20_online', m.to(device).eval()))
    return sources


# ==========================
# 3. 全局配置
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_TRIALS = 3
TEST_ADV_NUM = 1000
STANDARD_EPS = 8 / 255  # 消融实验使用标准扰动


# ==========================
# 4. 主实验逻辑
# ==========================
def run_ablation_study():
    setup_chinese_font()
    save_dir = PROJECT_ROOT / "result" / "ablation"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 模型配置
    model = DualDomainAE(latent_dim=8).to(DEVICE)
    ckpt_path = PROJECT_ROOT / "checkpoint" / "vmamba" / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    _, test_loader = get_loaders()
    print("🧪 正在初始化统计基准...")
    detector.init_threshold(model, test_loader, device=DEVICE)

    # 加载攻击源模型（供 generate_adv_dataset 内部使用，但该函数会自己加载，此处仅为预热缓存）
    _ = load_local_classifiers(DEVICE)

    # 生成或加载标准对抗样本文件
    print(f"🚀 正在准备高迁移对抗样本 (eps={STANDARD_EPS*255:.0f}/255)...")
    adv_file_path = generate_adv_dataset(
        need_count=TEST_ADV_NUM,
        use_train_set=False,
        force=True,          # 强制生成，确保样本一致性
        batch_size=32,
        eps=STANDARD_EPS
    )
    adv_data = torch.load(adv_file_path, map_location='cpu')
    norm_imgs = adv_data['norm_images'].to(DEVICE)
    adv_imgs = adv_data['adv_images'].to(DEVICE)

    # 初始化 plot_data，增加 phase 字段
    plot_data = {"norm": {"f": [], "s": [], "phase": []},
                 "adv": {"f": [], "s": [], "phase": []}}
    results = np.zeros((NUM_TRIALS, 3, 2))

    def get_metrics_custom(images, mode, is_adv=False):
        batch_size = 64
        preds = []
        f_raw_list, s_raw_list, phase_list = [], [], []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(DEVICE)
            with torch.no_grad():
                s_raw, f_raw = detector.get_raw_mse(model, batch, DEVICE)
                s_raw_list.extend(s_raw.cpu().tolist())
                f_raw_list.extend(f_raw.cpu().tolist())

                if mode == 2:   # 双域联合模式，收集相位数据
                    phase_raw = detector.get_phase_diff_score(model, batch, DEVICE)
                    phase_list.extend(phase_raw.cpu().tolist())

                if mode == 0:      # 仅空域
                    p = s_raw > detector.SPATIAL_MSE_THRESH
                elif mode == 1:    # 仅频域
                    p = f_raw > detector.FREQ_MSE_THRESH
                else:              # 双域联合（含相位）
                    p = detector.is_adversarial_batch(model, batch, DEVICE)
            preds.append(p.cpu())
        res = torch.cat(preds).float().mean().item() * 100
        if mode == 2:
            key = "adv" if is_adv else "norm"
            plot_data[key]["f"].extend(f_raw_list)
            plot_data[key]["s"].extend(s_raw_list)
            plot_data[key]["phase"].extend(phase_list)
        return res if is_adv else (100.0 - res)

    configs = ["仅空域检测器", "仅频域检测器", "DUAD"]
    for t in range(NUM_TRIALS):
        for m in range(3):
            results[t, m, 0] = get_metrics_custom(norm_imgs, m, is_adv=False)
            results[t, m, 1] = get_metrics_custom(adv_imgs, m, is_adv=True)
        print(f"✅ Trial {t + 1}/{NUM_TRIALS} 完成")

    # ==========================
    # 5. 绘图部分（增强版散点图）
    # ==========================
    def draw_scatter():
        f_thresh = detector.FREQ_MSE_THRESH
        s_thresh = detector.SPATIAL_MSE_THRESH
        phase_thresh = detector.PHASE_DIFF_THRESH

        # 提取数据
        f_norm = np.array(plot_data["norm"]["f"])
        s_norm = np.array(plot_data["norm"]["s"])
        f_adv = np.array(plot_data["adv"]["f"])
        s_adv = np.array(plot_data["adv"]["s"])
        phase_adv = np.array(plot_data["adv"]["phase"])

        # 判定是否位于左下角象限（空间和频域均未超过独立阈值）
        in_safe_quadrant = (s_adv <= s_thresh) & (f_adv <= f_thresh)

        # 判定相位联合捕获：位于安全象限内，且满足相位联合条件
        cond_phase_captured = in_safe_quadrant & (phase_adv > phase_thresh) & (s_adv > s_thresh * 0.6)

        # 其余对抗样本（不在安全象限内，或不满足相位条件的）
        cond_others = ~cond_phase_captured

        plt.figure(figsize=(10, 8))

        # 正常样本（蓝色）
        plt.scatter(f_norm, s_norm, c='#4C72B0', alpha=0.25, s=8, label='正常样本')

        # 未被相位补漏的对抗样本（红色）
        if np.any(cond_others):
            plt.scatter(f_adv[cond_others], s_adv[cond_others],
                        c='#C44E52', alpha=0.5, s=12, label='对抗 (未捕获)')

        # 被相位补漏捕获的对抗样本（仅安全象限内，橙色叉号）
        if np.any(cond_phase_captured):
            plt.scatter(f_adv[cond_phase_captured], s_adv[cond_phase_captured],
                        c='orange', marker='x', alpha=0.9, s=25,
                        label='对抗 (相位补漏捕获)')

        # 阈值线
        plt.axvline(x=f_thresh, color='green', ls='--', alpha=0.5, label='频域阈值')
        plt.axhline(y=s_thresh, color='orange', ls='--', alpha=0.5, label='空域阈值')

        plt.title("空域-频域误差分布与相位补漏机制")
        plt.xlabel("频域重建误差")
        plt.ylabel("空域重建误差")
        plt.legend(loc='upper right', markerscale=1.5)
        plt.tight_layout()
        plt.savefig(save_dir / "scatter_distribution_enhanced.png", dpi=300)
        plt.close()
        print(f"📊 相位补漏散点图已保存: {save_dir / 'scatter_distribution_enhanced.png'}")

    def draw_bar_chart():
        avg_results = np.mean(results, axis=0)
        labels = configs
        normal_rate = avg_results[:, 0]
        detect_rate = avg_results[:, 1]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width / 2, normal_rate, width, label='正常放行率 (TNR)', color='#55A868', alpha=0.8)
        rects2 = ax.bar(x + width / 2, detect_rate, width, label='对抗检测率 (TPR)', color='#C44E52', alpha=0.8)

        ax.set_ylabel('百分比 (%)')
        ax.set_title('不同检测模式性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 115)
        ax.legend(loc='upper right')

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.savefig(save_dir / "ablation_bar_chart.png", dpi=300)
        plt.close()

    draw_scatter()
    draw_bar_chart()

    print(f"\n📊 消融实验结果汇总:")
    print("=" * 100)
    print(f"{'配置模式':<25} | {'正常放行率':<15} | {'对抗检测率'}")
    print("-" * 100)
    for i in range(3):
        m_norm, m_adv = np.mean(results[:, i, 0]), np.mean(results[:, i, 1])
        print(f"{configs[i]:<25} | {m_norm:14.2f}% | {m_adv:12.2f}%")
    print("=" * 100)
    print(f"📈 柱状图已保存至: {save_dir / 'ablation_bar_chart.png'}")


if __name__ == "__main__":
    run_ablation_study()