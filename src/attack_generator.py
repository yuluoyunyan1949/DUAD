import os
import sys
import torch
import torch.nn.functional as F
import importlib.util
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from tqdm import tqdm

from src.data_loader import get_loaders

# -------------------- 核心架构配置 --------------------
SOURCE_MODEL_NAMES = ['resnet20', 'mobileNetV2', 'repvgg']
TARGET_MODEL_NAMES = ['vgg16_bn', 'shufflenetv2']

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

SAVE_DIR = './saved_adv_samples'
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- 从本地加载模型 --------------------
def load_model(model_name, device):
    class_name_mapping = {
        'mobileNetV2': 'cifar10_mobilenetv2_x1_0',
        'resnet20': 'cifar10_resnet20',
        'vgg16_bn': 'cifar10_vgg16_bn',
        'repvgg': 'cifar10_repvgg_a0',
        'shufflenetv2': 'cifar10_shufflenetv2_x1_0'
    }
    target_name = class_name_mapping[model_name]

    spec = importlib.util.spec_from_file_location(model_name, f"./checkpoint/{model_name}/model.py")
    module = importlib.util.module_from_spec(spec)

    sys.modules[model_name] = module

    spec.loader.exec_module(module)

    build_func = getattr(module, target_name)
    model = build_func()

    weight_path = f"./checkpoint/{model_name}/mfiae_model.pt"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model.to(device).eval()


# -------------------- DI-MI 集成攻击核心类 --------------------
class EnsembleDIMIFGSM:
    def __init__(self, models, eps, alpha, steps, device):
        self.models = models
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self.mean = CIFAR10_MEAN.to(device)
        self.std = CIFAR10_STD.to(device)

    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        adv = images.clone().detach().requires_grad_(True)
        momentum = torch.zeros_like(images)

        for _ in range(self.steps):
            adv_curr = adv.clone().detach().requires_grad_(True)

            if torch.rand(1) > 0.5:
                res = int(32 * 0.9)
                x_di = F.interpolate(adv_curr, size=(res, res), mode='bilinear')
                p = 32 - res
                pad_t, pad_l = torch.randint(0, p + 1, (1,)).item(), torch.randint(0, p + 1, (1,)).item()
                x_di = F.pad(x_di, [pad_l, p - pad_l, pad_t, p - pad_t])
            else:
                x_di = adv_curr

            x_norm = (x_di - self.mean) / self.std
            loss = sum(F.cross_entropy(m(x_norm), labels) for m in self.models) / len(self.models)
            grad = torch.autograd.grad(loss, adv_curr)[0]

            grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = 1.0 * momentum + grad

            adv = adv.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv - images, -self.eps, self.eps)
            adv = torch.clamp(images + delta, 0, 1).detach()
        return adv


# -------------------- 对外接口：生成高迁移对抗样本集 --------------------
import os
import sys
import torch
import torch.nn.functional as F
import importlib.util
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from tqdm import tqdm

from src.data_loader import get_loaders

# -------------------- 核心架构配置 --------------------
SOURCE_MODEL_NAMES = ['resnet20', 'mobileNetV2', 'repvgg']
TARGET_MODEL_NAMES = ['vgg16_bn', 'shufflenetv2']

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

SAVE_DIR = './saved_adv_samples'
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------- 从本地加载模型 --------------------
def load_model(model_name, device):
    class_name_mapping = {
        'mobileNetV2': 'cifar10_mobilenetv2_x1_0',
        'resnet20': 'cifar10_resnet20',
        'vgg16_bn': 'cifar10_vgg16_bn',
        'repvgg': 'cifar10_repvgg_a0',
        'shufflenetv2': 'cifar10_shufflenetv2_x1_0'
    }
    target_name = class_name_mapping[model_name]

    spec = importlib.util.spec_from_file_location(model_name, f"./checkpoint/{model_name}/model.py")
    module = importlib.util.module_from_spec(spec)

    sys.modules[model_name] = module
    spec.loader.exec_module(module)

    build_func = getattr(module, target_name)
    model = build_func()

    weight_path = f"./checkpoint/{model_name}/mfiae_model.pt"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model.to(device).eval()


# -------------------- DI-MI 集成攻击核心类 --------------------
class EnsembleDIMIFGSM:
    def __init__(self, models, eps, alpha, steps, device):
        self.models = models
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self.mean = CIFAR10_MEAN.to(device)
        self.std = CIFAR10_STD.to(device)

    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        adv = images.clone().detach().requires_grad_(True)
        momentum = torch.zeros_like(images)

        for _ in range(self.steps):
            adv_curr = adv.clone().detach().requires_grad_(True)

            if torch.rand(1) > 0.5:
                res = int(32 * 0.9)
                x_di = F.interpolate(adv_curr, size=(res, res), mode='bilinear')
                p = 32 - res
                pad_t, pad_l = torch.randint(0, p + 1, (1,)).item(), torch.randint(0, p + 1, (1,)).item()
                x_di = F.pad(x_di, [pad_l, p - pad_l, pad_t, p - pad_t])
            else:
                x_di = adv_curr

            x_norm = (x_di - self.mean) / self.std
            loss = sum(F.cross_entropy(m(x_norm), labels) for m in self.models) / len(self.models)
            grad = torch.autograd.grad(loss, adv_curr)[0]

            grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = 1.0 * momentum + grad

            adv = adv.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv - images, -self.eps, self.eps)
            adv = torch.clamp(images + delta, 0, 1).detach()
        return adv


# -------------------- 对外接口：生成高迁移对抗样本集 --------------------
def generate_adv_dataset(need_count=None, use_train_set=False, force=True, batch_size=32,
                         eps=8/255, steps=10, psnr_threshold=30.0):
    """
    从完整的 CIFAR-10 数据集中生成高迁移对抗样本并保存到文件。

    参数:
        need_count (int|None): 期望生成的对抗样本数量，None 表示尽可能多。
        use_train_set (bool): True 使用训练集，False 使用测试集。
        force (bool): 是否强制覆盖已有文件。
        batch_size (int): 攻击生成时每批处理的图像数量。
        eps (float): 扰动幅度，默认 8/255。
        steps (int): 攻击迭代步数，默认 10。
        psnr_threshold (float): 最低 PSNR 阈值，默认 30.0 dB。

    返回:
        str|None: 保存文件的路径，若未生成任何样本则返回 None。
    """
    split_name = "TRAIN" if use_train_set else "TEST"
    eps_int = int(round(eps * 255))
    count_str = "ALL" if need_count is None else str(need_count)
    filename = f"Adv_DI_MI_{split_name}_E{eps_int}_N{count_str}.pt"
    save_path = os.path.join(SAVE_DIR, filename)

    if os.path.exists(save_path) and not force:
        print(f"📦 已存在现有数据集: {save_path}")
        return save_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_models = [load_model(name, device) for name in SOURCE_MODEL_NAMES]
    target_models = [load_model(name, device) for name in TARGET_MODEL_NAMES]

    attacker = EnsembleDIMIFGSM(source_models, eps, eps / steps, steps, device)

    train_loader, test_loader = get_loaders(batch_size=batch_size, pin_memory=True)
    loader = train_loader if use_train_set else test_loader

    adv_list, norm_list, labels_list = [], [], []
    count = 0

    total_steps = need_count if need_count is not None else len(loader.dataset)
    desc = f"🚀 生成 {split_name} 对抗样本 (eps={eps*255:.1f}/255)"
    if need_count is None:
        desc += " (无上限模式)"
    else:
        desc += f" (目标 {need_count} 张)"
    pbar = tqdm(total=total_steps, desc=desc)

    for images, labels in loader:
        if need_count is not None and count >= need_count:
            break

        img_b, lbl_b = images.to(device), labels.to(device)
        adv_b = attacker(img_b, lbl_b)

        with torch.no_grad():
            x_norm = (adv_b - CIFAR10_MEAN.to(device)) / CIFAR10_STD.to(device)
            success_masks = [(m(x_norm).argmax(1) != lbl_b) for m in target_models]
            transfer_success = torch.stack(success_masks).all(dim=0)

        img_np = img_b.cpu().numpy()
        adv_np = adv_b.cpu().numpy()
        quality_mask = torch.tensor(
            [psnr_func(img_np[i], adv_np[i], data_range=1.0) >= psnr_threshold for i in range(len(img_np))],
            device=device
        )

        final_mask = transfer_success & quality_mask
        keep_idx = torch.where(final_mask)[0]

        if len(keep_idx) > 0:
            if need_count is not None:
                num = min(len(keep_idx), need_count - count)
            else:
                num = len(keep_idx)
            keep_idx = keep_idx[:num]
            norm_list.append(img_b[keep_idx].cpu())
            adv_list.append(adv_b[keep_idx].cpu())
            labels_list.append(lbl_b[keep_idx].cpu())
            count += num
            pbar.update(num if need_count is not None else len(keep_idx))

        torch.cuda.empty_cache()

        if need_count is None:
            pbar.update(len(images))

    pbar.close()

    if count == 0:
        print("❌ 错误：未找到任何合格样本。")
        return None

    data = {
        'norm_images': torch.cat(norm_list).clamp(0, 1),
        'adv_images': torch.cat(adv_list).clamp(0, 1),
        'labels': torch.cat(labels_list),
        'meta': {
            'attack': 'DI-MI Ensemble',
            'eps': eps,
            'steps': steps,
            'psnr_threshold': psnr_threshold,
            'need_count': need_count,
            'actual_count': count,
            'split': split_name,
        }
    }
    torch.save(data, save_path)
    print(f"💾 高迁移严选数据集已保存: {save_path} (实际生成 {count} 张)")
    return save_path