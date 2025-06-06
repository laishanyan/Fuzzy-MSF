import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel

from .fuzzy_embedding import FuzzyEmbedding
from .macl import MACL
from .SAP_Attention import SAPAttention
from .ViT import ViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Fuzzy_MM(nn.Module):
    def __init__(self,
                 img_feat_dim=768,
                 txt_feat_dim=768,
                 emo_dim=1,
                 proj_dim=768,
                 num_views=3,
                 fuzzy_clusters=5,
                 hidden_dim=768,
                 num_classes=3
                 ):
        super().__init__()
        self.fuzzy_img = FuzzyEmbedding(img_feat_dim, num_clusters=fuzzy_clusters, embed_dim=proj_dim)
        self.fuzzy_txt = FuzzyEmbedding(txt_feat_dim, num_clusters=fuzzy_clusters, embed_dim=proj_dim)
        self.macl = MACL(proj_dim, proj_dim, proj_dim, num_views=num_views)
        self.sap = SAPAttention(proj_dim, emo_dim, hidden_dim)
        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img_feat, txt_feat, emo_feat, labels=None):
        # 1. 模糊语义嵌入
        fuzzy_img_embed, img_soft_assign, img_entropy = self.fuzzy_img(img_feat)
        fuzzy_txt_embed, txt_soft_assign, txt_entropy = self.fuzzy_txt(txt_feat)

        # 2. 多视角对齐损失
        macl_loss = self.macl(fuzzy_img_embed, fuzzy_txt_embed)

        # 3. 主观性感知融合
        fused_feat = self.sap(fuzzy_img_embed, fuzzy_txt_embed, emo_feat)

        return fused_feat, macl_loss

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc1 = nn.Linear(config.dim_model*(config.pad_size+1), config.num_classes)
        self.vit = ViT(config)
        self.dropout = nn.Dropout(config.dropout)
        self.fuzzy_mm = Fuzzy_MM().to(device)


    def forward(self, x):
        img = x[0]
        context = x[1]  # 输入的句子
        mask = x[3]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        text, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        img = self.vit(img)
        out, macloss= self.fuzzy_mm(img, text, text_cls)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.fc1(out)
        out = self.dropout(out)
        return out, macloss