import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDOneClassSVM

class OCSVM:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=16, batch_size=256)
        self.model = SGDOneClassSVM(nu=0.05, random_state=42)

    def get_latent(self, model, x):
        with torch.no_grad():
            _, s_feat, _ = model(x)
            return s_feat.flatten(1).cpu().numpy()

    def fit(self, model, loader):
        # 第一步：流式计算标准化参数（均值和方差）
        print("⏳ 计算特征标准化参数...")
        sum_x, sum_x2, total = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.cuda()
            batch = self.get_latent(model, x)
            sum_x += batch.sum(axis=0)
            sum_x2 += (batch ** 2).sum(axis=0)
            total += batch.shape[0]
        mean = sum_x / total
        std = np.sqrt(sum_x2 / total - mean ** 2) + 1e-8
        self.scaler.mean_ = mean
        self.scaler.scale_ = std

        # 第二步：分批执行增量 PCA
        print("⏳ 执行增量 PCA...")
        for x, _ in loader:
            x = x.cuda()
            batch = self.get_latent(model, x)
            batch_scaled = (batch - mean) / std
            self.pca.partial_fit(batch_scaled)
            torch.cuda.empty_cache()

        # 第三步：分批转换数据并训练 SGDOneClassSVM
        print("⏳ 训练 SGDOneClassSVM...")
        for x, _ in loader:
            x = x.cuda()
            batch = self.get_latent(model, x)
            batch_scaled = (batch - mean) / std
            batch_pca = self.pca.transform(batch_scaled)
            self.model.partial_fit(batch_pca)
            torch.cuda.empty_cache()

        print("✅ OC-SVM 增量训练完成（使用全部训练集）")