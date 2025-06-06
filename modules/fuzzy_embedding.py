import torch
import torch.nn as nn
import torch.nn.functional as F


class FuzzyEmbedding(nn.Module):
    def __init__(self, input_dim, num_clusters=5, embed_dim=256, temperature=0.1):
        """
        input_dim: 输入特征维度（图像或文本）
        num_clusters: 模糊语义中心数量
        embed_dim: 中心向量维度
        temperature: 控制软分配的平滑度
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature

        # 模糊中心向量，形状 (num_clusters, embed_dim)
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, embed_dim))

        # 输入特征到embed_dim的线性映射
        self.embedding_proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """
        x: 输入特征 (B, input_dim)
        返回:
          fuzzy_embed: (B, embed_dim) -- 模糊嵌入向量（加权中心）
          soft_assign: (B, num_clusters) -- 样本对各中心的归属概率（软分配）
          entropy_loss: 熵正则项，鼓励软分配
        """
        B = x.size(0)
        x_embed = self.embedding_proj(x)  # (B, embed_dim)
        x_embed = F.normalize(x_embed, dim=-1)  # 归一化

        # 计算样本与各中心的余弦相似度 (B, num_clusters)
        centers_norm = F.normalize(self.cluster_centers, dim=-1)
        sim_matrix = torch.matmul(x_embed, centers_norm.t())  # (B, num_clusters)

        # 软分配概率
        soft_assign = F.softmax(sim_matrix / self.temperature, dim=-1)  # (B, num_clusters)

        # 模糊嵌入：软分配加权的中心向量和
        fuzzy_embed = torch.matmul(soft_assign, self.cluster_centers)  # (B, embed_dim)
        fuzzy_embed = F.normalize(fuzzy_embed, dim=-1)

        # 熵正则项，鼓励soft分配均匀分布，防止过于确定
        entropy = - (soft_assign * torch.log(soft_assign + 1e-8)).sum(dim=1).mean()

        return fuzzy_embed, soft_assign, entropy
