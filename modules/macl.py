import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim, num_views=3):
        super().__init__()
        self.pad_size = 64
        self.num_views = num_views
        self.proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.pad_size*input_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
            )
            for _ in range(num_views)
        ])

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        # 输出形状: (batch_size, num_views, proj_dim)
        projections = [proj(x).unsqueeze(1) for proj in self.proj_heads]
        return torch.cat(projections, dim=1)


class MACL(nn.Module):
    def __init__(self, img_feat_dim=768, txt_feat_dim=768, proj_dim=768, num_views=3, temperature=0.07):
        super().__init__()
        self.img_proj = MultiViewProjectionHead(img_feat_dim, proj_dim, num_views)
        self.txt_proj = MultiViewProjectionHead(txt_feat_dim, proj_dim, num_views)
        self.temperature = temperature
        self.num_views = num_views

    def local_global_matching_loss(self, img_proj, txt_proj):
        """
        img_proj, txt_proj: shape (batch_size, num_views, proj_dim)
        """
        B, V, D = img_proj.shape
        losses = []
        for v in range(V):
            # 单个视角
            img_v = F.normalize(img_proj[:, v, :], dim=-1)
            txt_v = F.normalize(txt_proj[:, v, :], dim=-1)

            # 计算相似度矩阵 (B, B)
            logits = torch.matmul(img_v, txt_v.T) / self.temperature
            labels = torch.arange(B).to(img_v.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            losses.append((loss_i2t + loss_t2i) / 2.0)

        return torch.stack(losses).mean()  # 所有视角平均

    def forward(self, img_feats, txt_feats):
        """
        img_feats: (batch_size, img_feat_dim)
        txt_feats: (batch_size, txt_feat_dim)
        """
        img_proj = self.img_proj(img_feats)  # (B, V, D)
        txt_proj = self.txt_proj(txt_feats)
        loss = self.local_global_matching_loss(img_proj, txt_proj)/2
        return loss
