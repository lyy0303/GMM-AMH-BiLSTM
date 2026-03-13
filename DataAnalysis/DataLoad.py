import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import torch

class ECGDataset(Dataset):
    def __init__(self, features, labels, scaler=None):
        # 标准化特征（保留原有逻辑）
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(features)
        self.labels = labels.astype(int).values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # LSTM需要[序列长度, 特征数]，reshape为(1, 16)（单时间步）
        feature = torch.tensor(self.features[idx], dtype=torch.float32).reshape(1, -1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # 直接用原始标签
        return feature, label


def load_ecg_data(batch_size=32):
    # 直接读取已划分好的训练/验证/测试集
    train_df = pd.read_csv('D:/PycharmProjects/EcgGmm/DataAnalysis/Data_of_train_and_test_VG/train_df_SMOTE.csv')
    # train_df = pd.read_csv('D:/PycharmProjects/EcgGmm/DataAnalysis/Data_of_train_and_test_VG/train_df.csv')
    val_df = pd.read_csv('D:/PycharmProjects/EcgGmm/DataAnalysis/Data_of_train_and_test_VG/val_df.csv')
    test_df = pd.read_csv('D:/PycharmProjects/EcgGmm/DataAnalysis/Data_of_train_and_test_VG/test_df.csv')

    # train_feat = train_df.iloc[:, 3:19]  # 第4列=索引3，第19列=索引18，iloc是左闭右开，所以取3:19
    train_feat = train_df.iloc[:, :16]  # 采样用这个
    train_label = train_df.iloc[:, -1]  # 最后一列作为标签（已为0-4）

    val_feat = val_df.iloc[:, 3:19]
    val_label = val_df.iloc[:, -1]

    test_feat = test_df.iloc[:, 3:19]
    test_label = test_df.iloc[:, -1]

    # 构建数据集（训练集拟合scaler，验证/测试集复用）
    train_dataset = ECGDataset(train_feat, train_label)
    val_dataset = ECGDataset(val_feat, val_label, scaler=train_dataset.scaler)
    test_dataset = ECGDataset(test_feat, test_label, scaler=train_dataset.scaler)

    # 构建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(train_loader)

    # 返回加载器（若需要标签映射，可返回空字典或直接删除）
    return train_loader, val_loader, test_loader