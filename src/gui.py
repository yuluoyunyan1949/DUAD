import gradio as gr
import torch
import numpy as np
from PIL import Image
import random
import os
import glob
import torchvision.transforms as transforms

# 导入核心模块
from src.encoder import DualDomainAE
import src.detector as detector
from src.data_loader import get_loaders

# -------------------- 全局初始化 --------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "./checkpoint/vmamba/best_model.pth"

# 反归一化参数
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

CURRENT_SAMPLES_TENSORS = []

# 图像预处理：调整大小、转张量、归一化到[0,1]
transform_upload = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

def get_adv_file():
    """固定返回训练集高迁移对抗样本文件，若缺失则回退到任意可用文件"""
    fixed_path = './saved_adv_samples/Adv_DI_MI_TRAIN_E8_N20000.pt'
    if os.path.exists(fixed_path):
        return fixed_path
    else:
        # 回退：查找任意 DI_MI 文件
        list_of_files = glob.glob('./saved_adv_samples/Adv_DI_MI*.pt')
        if list_of_files:
            return max(list_of_files, key=os.path.getctime)
        else:
            return None


# 模型载入
model = DualDomainAE(latent_dim=8).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# 阈值初始化
train_loader, _ = get_loaders(batch_size=128)
detector.init_threshold(model, train_loader, DEVICE)

# 数据加载（用于采样池）
ADV_DATASET_PATH = get_adv_file()
NORM_IMAGES, ADV_IMAGES = None, None

if ADV_DATASET_PATH:
    try:
        adv_data = torch.load(ADV_DATASET_PATH, map_location='cpu', weights_only=False)
        NORM_IMAGES = adv_data.get('norm_images')
        ADV_IMAGES = adv_data.get('adv_images')
        print(f"✅ [GUI] 载入数据集: {os.path.basename(ADV_DATASET_PATH)}")
    except Exception as e:
        print(f"❌ [GUI] 加载错误: {e}")

if NORM_IMAGES is None:
    NORM_IMAGES = torch.rand(10, 3, 32, 32)
    ADV_IMAGES = torch.rand(10, 3, 32, 32)


# -------------------- 图像渲染函数 --------------------
def tensor_to_pil(img_tensor):
    img = img_tensor.detach().cpu().clone()
    if img.min() < -0.01 or img.max() > 1.01:
        img = img * CIFAR_STD + CIFAR_MEAN
    img = torch.clamp(img, 0, 1)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    pil_img = pil_img.resize((128, 128), resample=Image.LANCZOS)
    return pil_img


# -------------------- 采样池相关函数 --------------------
def handle_refresh_samples():
    global CURRENT_SAMPLES_TENSORS
    random.seed(None)
    num = min(5, len(NORM_IMAGES))
    indices = random.sample(range(len(NORM_IMAGES)), num)

    display_list = []
    CURRENT_SAMPLES_TENSORS = []

    for i in indices:
        display_list.append((tensor_to_pil(NORM_IMAGES[i]), f"🟢 正常 #{i}"))
        CURRENT_SAMPLES_TENSORS.append(NORM_IMAGES[i])
    for i in indices:
        display_list.append((tensor_to_pil(ADV_IMAGES[i]), f"🔴 攻击 #{i}"))
        CURRENT_SAMPLES_TENSORS.append(ADV_IMAGES[i])

    return display_list, "点击图片开始监测", ""


def detect_logic(evt: gr.SelectData):
    idx = evt.index
    if idx >= len(CURRENT_SAMPLES_TENSORS): return "Error", "Invalid Index"

    img_t = CURRENT_SAMPLES_TENSORS[idx].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        rec_orig, _, _ = model(img_t)
        spatial_mse = torch.mean((rec_orig - img_t) ** 2).item()
        amp_rec = detector.get_amp_spectrum(rec_orig)
        amp_x = detector.get_amp_spectrum(img_t)
        freq_mse = torch.mean((amp_rec - amp_x) ** 2).item()
        is_adv = bool(detector.is_adversarial_batch(model, img_t, DEVICE)[0])

    res_icon = "🚨 [Detected]" if is_adv else "✅ [Normal]"
    status_spatial = " > " if spatial_mse > detector.SPATIAL_MSE_THRESH else " <= "
    status_freq = " > " if freq_mse > detector.FREQ_MSE_THRESH else " <= "

    detail = (
        f"空间误差：{spatial_mse:.6f} {status_spatial} 空间误差阈值：{detector.SPATIAL_MSE_THRESH:.6f}\n"
        f"频域误差：{freq_mse:.6f} {status_freq} 频域误差阈值：{detector.FREQ_MSE_THRESH:.6f}\n"
    )
    return res_icon, detail


# -------------------- 新增：本地图像上传检测 --------------------
def detect_uploaded_image(image):
    """处理用户上传的图像，返回检测结果和详细信息"""
    if image is None:
        return "请上传图像", "未检测到图像"

    # 预处理：PIL -> Tensor (3,32,32) 值域[0,1]
    img_tensor = transform_upload(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        rec_orig, _, _ = model(img_tensor)
        spatial_mse = torch.mean((rec_orig - img_tensor) ** 2).item()
        amp_rec = detector.get_amp_spectrum(rec_orig)
        amp_x = detector.get_amp_spectrum(img_tensor)
        freq_mse = torch.mean((amp_rec - amp_x) ** 2).item()
        is_adv = bool(detector.is_adversarial_batch(model, img_tensor, DEVICE)[0])

    res_icon = "🚨 [Adversarial]" if is_adv else "✅ [Normal]"
    status_spatial = " > " if spatial_mse > detector.SPATIAL_MSE_THRESH else " <= "
    status_freq = " > " if freq_mse > detector.FREQ_MSE_THRESH else " <= "

    detail = (
        f"空间误差：{spatial_mse:.6f} {status_spatial} 空间误差阈值：{detector.SPATIAL_MSE_THRESH:.6f}\n"
        f"频域误差：{freq_mse:.6f} {status_freq} 频域误差阈值：{detector.FREQ_MSE_THRESH:.6f}\n"
    )
    return res_icon, detail


# -------------------- UI 布局 --------------------
CSS = """
.gr-gallery img { 
    image-rendering: auto; 
    border-radius: 12px;
}
"""

def gui_run():
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# DUAD 对抗样本监测台")

        with gr.Tab("演示样本检测"):
            gallery = gr.Gallery(label="采样池", columns=5, rows=2, height="auto")
            btn_refresh = gr.Button("刷新样本", variant="primary")
            with gr.Row():
                out_res_pool = gr.Textbox(label="状态")
                out_det_pool = gr.Textbox(label="检测结论", lines=6)
            btn_refresh.click(handle_refresh_samples, outputs=[gallery, out_res_pool, out_det_pool])
            gallery.select(detect_logic, outputs=[out_res_pool, out_det_pool])
            demo.load(handle_refresh_samples, outputs=[gallery, out_res_pool, out_det_pool])

        with gr.Tab("用户样本检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_img = gr.Image(type="pil", label="上传图像 (将自动缩放至32x32)")
                    btn_upload = gr.Button("检测图像", variant="primary")
                with gr.Column(scale=2):
                    out_res_upload = gr.Textbox(label="检测结果")
                    out_det_upload = gr.Textbox(label="详细信息", lines=6)
            btn_upload.click(detect_uploaded_image, inputs=upload_img, outputs=[out_res_upload, out_det_upload])

    demo.launch(server_port=7860)