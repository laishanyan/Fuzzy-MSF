import torch
import torch.nn as nn
import torch.nn.functional as F

class SAPAttention(nn.Module):
    def __init__(self, feat_dim, emo_dim=1, hidden_dim=768):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, img_feat, txt_feat, emo_feat):
        """
        img_feat: (B, D) - 图像特征向量
        txt_feat: (B, D) - 文本特征向量
        emo_feat: (B, 3) - 情感向量（Bert情感分类的类别向量）
        """
        B, n, D = img_feat.shape
        # 构建查询、键、值
        emo_feat = emo_feat.view([B,1,D])
        fused_txt_emo = torch.cat([txt_feat, emo_feat], dim=1)
        fused_img_emo = torch.cat([img_feat, emo_feat], dim=1)

        query = self.query_proj(fused_img_emo)  # 以图像为查询
        key = self.key_proj(fused_txt_emo)  # 文本+情感作为键
        value = self.value_proj(fused_txt_emo)  # 同上

        # 注意力打分（B, 1）
        attn_scores = torch.matmul(query, key.permute(0, 2, 1))/ (key.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, 1, 1)

        # 加权融合
        attn_output = torch.matmul(attn_weights, value)  # (B, hidden_dim)
        output = self.output_proj(attn_output.squeeze(1))  # (B, D)

        # 残差连接 + LayerNorm
        fused = self.norm(fused_img_emo + output)
        return fused
