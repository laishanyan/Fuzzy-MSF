import torch
from pytorch_pretrained import BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'Fuzzy-MSF'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/val.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.image_path = dataset + '/data/images/'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 50                                            # epoch数
        self.batch_size = 8                                             # mini-batch大小
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1

        self.dim_model = 768
        self.hidden = 768
        self.last_hidden = 512
        self.num_head = 4
        self.num_encoder = 1
        self.embed = 768