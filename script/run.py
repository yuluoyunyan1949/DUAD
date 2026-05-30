import warnings
import os
import sys
import torch
from pathlib import Path

# --- 1. 屏蔽干扰日志 ---
warnings.filterwarnings("ignore")
os.environ["TORCH_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["SELECTIVE_SCAN_BACKEND"] = "oflex"

# --- 2. 环境挂载 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VMAMBA_PATH = PROJECT_ROOT / "VMamba"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VMAMBA_PATH))

from src.trainer import train_model          # 修正后的训练器
from src.attack_generator import generate_adv_dataset
from src.gui import gui_run

# --- 3. 核心配置 ---
CHECKPOINT_DIR = "./checkpoint/vmamba"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
S1_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "spatial_only_best.pth")
ADV_DATA_DIR = "./saved_adv_samples"


def prepare_env():
    """环境检查与 GPU 初始化"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✅ GPU 就绪: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    return device


if __name__ == "__main__":
    device = prepare_env()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---------- 步骤 1：训练防御模型 ----------
    print("\n【步骤 1/3】训练防御模型")
    if not os.path.exists(MODEL_SAVE_PATH):
        print("  >>> 未检测到完整模型权重，开始自动训练...")
        train_model(device=device, save_dir=CHECKPOINT_DIR)
        print("  >>> [完成] 训练任务结束。")
    else:
        print(f"  >>> [跳过] 使用现有权重: {MODEL_SAVE_PATH}")

    # ---------- 步骤 2：生成对抗样本 ----------
    print("\n【步骤 2/3】生成高迁移对抗样本 (用于 GUI 演示)...")
    os.makedirs(ADV_DATA_DIR, exist_ok=True)
    adv_file = os.path.join(ADV_DATA_DIR, "Adv_DI_MI_TEST_E8_N1000.pt")

    if not os.path.exists(adv_file):
        print("  >>> [生成] 正在生成 1000 张 DI-MI 测试集对抗样本...")
        generate_adv_dataset(need_count=1000, use_train_set=False, force=False)
        print("  >>> [完成] 对抗样本生成完毕。")
    else:
        print(f"  >>> [状态] 已存在对抗样本: {os.path.basename(adv_file)}")

    # ---------- 步骤 3：交互界面 ----------
    print("\n【步骤 3/3】启动 GUI...")
    try:
        gui_run()
    except KeyboardInterrupt:
        print("\n👋 用户终止运行。")
    except Exception as e:
        print(f"❌ 运行异常: {e}")

    print("\n" + "=" * 60)
    print("      🚀 DUAD 系统任务关闭。")
    print("=" * 60)