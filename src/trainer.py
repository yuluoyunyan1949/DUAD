import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import gc

from src.data_loader import get_loaders
from src.encoder import DualDomainAE
from src.attack_generator import generate_adv_dataset


# -------------------- 相位一致性损失 --------------------
def phase_consistency_loss(rec, target):
    # 强制转为 float32
    rec_f = rec.float()
    target_f = target.float()
    def get_phase(img):
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        fft = torch.fft.fft2(gray, dim=(-2, -1), norm='ortho')
        fft_shifted = torch.fft.fftshift(fft)
        return torch.angle(fft_shifted)

    phase_rec = get_phase(rec_f)
    phase_target = get_phase(target_f)
    diff = phase_rec - phase_target
    diff = torch.clamp(diff, -3.1416, 3.1416)
    loss = 1 - torch.cos(diff)
    return loss.mean()


# -------------------- 训练引擎（修正版：过滤解码器权重）--------------------
def train_model(device='cuda', save_dir='./checkpoint/vmamba',
                s1_epochs=15, s2_epochs=50, batch_size=64, latent_dim=8,
                freq_weight=0.1, phase_weight=0.05, lr=1e-4, weight_decay=0.05,  # lr 5e-4 → 1e-4
                auto_gen_adv=True):
    """
    对抗净化自编码器训练。

    参数:
        auto_gen_adv: 若训练集对抗样本不存在，是否自动生成（默认 True）。
    """
    cfg = {
        'lr': lr,
        'batch_size': batch_size,
        's1_epochs': s1_epochs,
        's2_epochs': s2_epochs,
        'latent_dim': latent_dim,
        'freq_weight': freq_weight,
        'phase_weight': phase_weight,
        'weight_decay': weight_decay
    }

    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 干净数据加载器 ----------
    train_loader_clean, val_loader_clean = get_loaders(batch_size=cfg['batch_size'])
    scaler = GradScaler('cuda')
    mse_criterion = nn.MSELoss()

    def freq_mse_loss(rec, x):
        # 强制转为 float32，避免混合精度下 FFT 的不稳定性
        rec_f = rec.float()
        x_f = x.float()
        amp_rec = torch.log1p(torch.abs(torch.fft.fft2(rec_f, dim=(-2, -1), norm='ortho')).clamp_min(1e-8))
        amp_x = torch.log1p(torch.abs(torch.fft.fft2(x_f, dim=(-2, -1), norm='ortho')).clamp_min(1e-8))
        return mse_criterion(amp_rec, amp_x)

    # ==================== 阶段 1: 空间域预训练 ====================
    s1_path = os.path.join(save_dir, 'spatial_only_best.pth')
    if not os.path.exists(s1_path):
        print(f"\n>>> [S1] 启动空域预训练 (干净图像 + 高斯噪声 → 干净图像)")
        model_s1 = DualDomainAE(
            latent_dim=cfg['latent_dim'],
            use_freq_branch=False
        ).to(device)
        opt_s1 = torch.optim.AdamW(model_s1.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        for epoch in range(cfg['s1_epochs']):
            model_s1.train()
            losses = []
            for images, _ in train_loader_clean:
                images = images.to(device, non_blocking=True)
                opt_s1.zero_grad(set_to_none=True)
                with autocast('cuda'):
                    noisy = torch.clamp(images + 0.05 * torch.randn_like(images), 0, 1)
                    rec, _, _ = model_s1(noisy)
                    loss = mse_criterion(rec, images)
                scaler.scale(loss).backward()
                scaler.step(opt_s1)
                scaler.update()
                losses.append(loss.item())

            avg_l = np.mean(losses)
            print(f"  Ep {epoch + 1:02d} | MSE: {avg_l:.5f} | PSNR: {10 * np.log10(1 / avg_l):.2f}dB")
            gc.collect()
            torch.cuda.empty_cache()

        torch.save(model_s1.state_dict(), s1_path)
        del model_s1, opt_s1
        gc.collect()
        torch.cuda.empty_cache()

    # ==================== 阶段 2: 对抗净化训练 ====================
    print(f"\n>>> [S2] 启动对抗净化训练 (对抗样本 → 干净图像)")
    print(f"      损失: MSE + {cfg['freq_weight']}*FreqMSE + {cfg['phase_weight']}*PhaseLoss")

    # 确保训练集对抗样本存在
    adv_train_path = './saved_adv_samples/Adv_DI_MI_TRAIN_E8_N20000.pt'
    if not os.path.exists(adv_train_path):
        if auto_gen_adv:
            print("⚠️ 训练集对抗样本不存在，正在自动生成 (need_count=20000, use_train_set=True)...")
            generate_adv_dataset(need_count=20000, use_train_set=True, force=False)
        else:
            raise FileNotFoundError(f"对抗样本训练集不存在: {adv_train_path}")

    adv_data = torch.load(adv_train_path, map_location='cpu')
    adv_images = adv_data['adv_images']
    norm_images = adv_data['norm_images']

    paired_dataset = torch.utils.data.TensorDataset(adv_images, norm_images)
    paired_loader = torch.utils.data.DataLoader(
        paired_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # 准备一小批对抗样本用于验证净化效果
    val_adv_path = './saved_adv_samples/Adv_DI_MI_TEST_E8_N1000.pt'
    if os.path.exists(val_adv_path):
        val_adv_data = torch.load(val_adv_path, map_location='cpu')
        val_adv_imgs = val_adv_data['adv_images'][:200].to(device)  # 取200张
        val_adv_clean = val_adv_data['norm_images'][:200].to(device)
    else:
        val_adv_imgs = None

    model = DualDomainAE(
        latent_dim=cfg['latent_dim'],
        use_freq_branch=True,
        freq_input_channels=3
    ).to(device)

    # ---------- 关键修正：加载第一阶段权重时过滤解码器 ----------
    if os.path.exists(s1_path):
        s1_state = torch.load(s1_path, map_location=device)
        # 排除解码器 (dec.) 和频域分支 (freq_enc) 相关的键，因为这些部分结构与第二阶段不兼容
        filtered_state = {
            k: v for k, v in s1_state.items()
            if not (k.startswith('dec.') or k.startswith('freq_enc.'))
        }
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        if missing_keys:
            # 预期缺失：解码器、频域分支权重，以及可能的 stem 差异（第一阶段 stem 可能有 BN，第二阶段无）
            print(f"ℹ️  [映射] 空间编码器权重已加载，解码器/频域分支将随机初始化 (缺失键: {len(missing_keys)} 个)")
        else:
            print("✅ [映射] 空间编码器权重已完全加载")
    else:
        print("⚠️ 未找到第一阶段权重，从头开始训练")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['s2_epochs'])
    best_score = float('inf')

    for epoch in range(cfg['s2_epochs']):
        model.train()
        stats = {'mse': [], 'fmse': [], 'phase': []}

        for adv_batch, clean_batch in paired_loader:
            adv_batch = adv_batch.to(device, non_blocking=True)
            clean_batch = clean_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                rec, _, _ = model(adv_batch)
                l_mse = mse_criterion(rec, clean_batch)
                l_fmse = freq_mse_loss(rec, clean_batch)
                l_phase = phase_consistency_loss(rec, clean_batch)
                total_loss = l_mse + cfg['freq_weight'] * l_fmse + cfg['phase_weight'] * l_phase

            # 新增：检测 nan/inf，跳过坏批次
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️ 警告：检测到 nan/inf 损失，跳过当前批次")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)  # 新增：解除缩放以便裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"⚠️ 警告：梯度爆炸/消失 (norm={grad_norm})，跳过本次更新")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()  # 必须调用 update 以保持 scaler 状态正确
                continue
            scaler.step(optimizer)
            scaler.update()

            stats['mse'].append(l_mse.item())
            stats['fmse'].append(l_fmse.item())
            stats['phase'].append(l_phase.item())

        # ---------- 验证 ----------
        model.eval()
        val_clean_mse = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader_clean:
                imgs = imgs.to(device)
                rec, _, _ = model(imgs)
                val_clean_mse += mse_criterion(rec, imgs).item()
        val_clean_mse /= len(val_loader_clean)

        val_adv_mse = 0.0
        if val_adv_imgs is not None:
            with torch.no_grad():
                rec_adv, _, _ = model(val_adv_imgs)
                val_adv_mse = mse_criterion(rec_adv, val_adv_clean).item()

        avg_mse = np.mean(stats['mse'])
        avg_fmse = np.mean(stats['fmse'])
        avg_phase = np.mean(stats['phase'])
        print(f"  ✨ Ep {epoch + 1:02d} | MSE: {avg_mse:.5f} | FMSE: {avg_fmse:.5f} | Phase: {avg_phase:.5f}")
        print(f"      Val Clean MSE: {val_clean_mse:.6f} | Val Adv MSE: {val_adv_mse:.6f}")

        # 综合分数：干净图像重建质量 + 对抗净化能力
        score = val_clean_mse + 0.5 * val_adv_mse
        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"  >>> [SAVE] 最佳模型已更新 (综合分数: {score:.6f})")

        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()

    return model