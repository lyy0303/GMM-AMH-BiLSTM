import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import torch

class ECGDataset(Dataset):
    def __init__(self, features, labels, scaler=None):
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
        feature = torch.tensor(self.features[idx], dtype=torch.float32).reshape(1, -1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label


def load_ecg_data(batch_size=32):
    train_df = pd.read_csv('./Data_of_train_and_test_VG/train_df_SMOTE.csv')
    # train_df = pd.read_csv('D:/PycharmProjects/EcgGmm/DataAnalysis/Data_of_train_and_test_VG/train_df.csv')
    val_df = pd.read_csv('./Data_of_train_and_test_VG/val_df.csv')
    test_df = pd.read_csv('./Data_of_train_and_test_VG/test_df.csv')

    # train_feat = train_df.iloc[:, 3:19]  # no SMOTE
    train_feat = train_df.iloc[:, :16]  # if SMOTE
    train_label = train_df.iloc[:, -1]

    val_feat = val_df.iloc[:, 3:19]
    val_label = val_df.iloc[:, -1]

    test_feat = test_df.iloc[:, 3:19]
    test_label = test_df.iloc[:, -1]

    train_dataset = ECGDataset(train_feat, train_label)
    val_dataset = ECGDataset(val_feat, val_label, scaler=train_dataset.scaler)
    test_dataset = ECGDataset(test_feat, test_label, scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(train_loader)

    return train_loader, val_loader, test_loader