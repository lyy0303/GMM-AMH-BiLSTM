import os
import numpy as np
import  pandas as pd
import ast
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

file_name = ['100', '101', '102', '103', '104', '105', '106', '107',
             '108', '109', '111', '112', '113', '114', '115', '116',
             '117', '118', '119', '121', '122', '123', '124', '200',
             '201', '202', '203', '205', '207', '208', '209', '210',
             '212', '213', '214', '215', '217', '219', '220', '221',
             '222', '223', '228', '230', '231', '232', '233', '234']


class GaussianData():
    def __init__(self, name: str = None):
        self.name = name
        self.path = f'D:/PycharmProjects/EcgGmm/GaussECG/signal_and_params_of_5_gaussian_VG/{name}/' if name else None

    def read_csv(self, path, file_name):
        """读取单个CSV文件"""
        df = pd.read_csv(os.path.join(path, file_name))
        for col in df.columns:
            if df[col].dtype == object and df[col].str.startswith('[').any():
                df[col] = df[col].apply(ast.literal_eval)
        return df

    def combine_single(self, name):
        """整合单个文件的数据"""
        path = f'D:/PycharmProjects/EcgGmm/GaussECG/signal_and_params_of_5_gaussian_VG/{name}/'
        df1 = self.read_csv(path, file_name='fitting_results.csv')
        df2 = self.read_csv(path, file_name='gaussian_params.csv')
        df2['beat_index'] = df1['beat_index']
        df2['record_name'] = df1['record_name']
        df2['symbol'] = df1['symbol']
        return df2

    def combine_all(self):
        """整合所有文件的数据"""
        all_data = []
        for name in file_name:
            print(f"Processing {name}...")
            df = self.combine_single(name)
            all_data.append(df)
        combined_df = pd.concat(all_data, ignore_index=True)
        new_columns = (['beat_index', 'record_name', 'symbol'] +
                       [col for col in combined_df.columns
                        if col not in ['beat_index', 'record_name', 'symbol']])

        combined_df = combined_df.reindex(columns=new_columns)
        combined_df.to_csv('D:/PycharmProjects/EcgGmm/DataAnalysis/Data_of_train_and_test_VG/combine_all.csv',index=False)
        return combined_df

    def class_type(self):
        df = self.combine_all()
        PEAK_SYMBOL = {
            '0': ['N', 'L', 'R', 'e', 'j'],   # N
            '1': ['A', 'a', 'J', 'S'],        # S
            '2': ['V', 'E'],                  # V
            '3': ['F'],                       # F
            '4': ['/', 'f', 'Q']              # Q
        }
        class_symbol = {}
        for type, category in PEAK_SYMBOL.items():
            for symbol in category:
                class_symbol[symbol] = type
        df['class_symbol'] = df['symbol'].map(class_symbol)
        df.to_csv('D:/PycharmProjects/EcgGmm/DataAnalysis/Data_of_train_and_test_VG/combine_all_class.csv',index=False)
        return df

    def partition_all(self, df):
        """对整合后的数据进行划分"""
        # df = self.class_type()
        train_df, internal = train_test_split(df, train_size=0.6, random_state=42)
        val_df, test_df = train_test_split(internal, train_size=0.5, random_state=42)
        train_df.to_csv('Data_of_train_and_test_VG/train_df.csv', index=False)
        val_df.to_csv('Data_of_train_and_test_VG/val_df.csv', index=False)
        test_df.to_csv('Data_of_train_and_test_VG/test_df.csv', index=False)
        print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")
        print(f"Training set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")
        return train_df, val_df, test_df

    def SMOTE_train(self):
        train_df = self.read_csv(r'D:/PycharmProjects/EcgGmm/DataAnalysis/Data_of_train_and_test_VG', 'train_df.csv')
        train_df_x =  train_df.iloc[:, 3:19]
        train_df_y = train_df.iloc[:, -1]
        smt = SMOTE(sampling_strategy={1: 5000, 2: 6000, 3:2000, 4: 8000}, random_state=42)
        train_df_x, train_df_y = smt.fit_resample(train_df_x, train_df_y)
        train_df_1 = pd.concat([train_df_x, train_df_y], axis=1)
        # label_0_sample = (train_df[train_df.iloc[:, -1] == 0].sample(n=min(20000, (train_df.iloc[:, -1] == 0).sum()), random_state=42))
        # train_df_SMOTE =  pd.concat([label_0_sample, train_df[train_df.iloc[:, -1] != 0]])
        train_df_1.to_csv('Data_of_train_and_test_VG/train_df_SMOTE_1.csv', index=False)
        # train_df_1.to_csv('Data_of_train_and_test_VG_0/train_df_SMOTE_1234.csv', index=False)
        # print('各个类别的数量', train_df_SMOTE['class_symbol'].value_counts())


if __name__ == "__main__":
    GD = GaussianData()
    # GD.class_type()
    # df=GD.read_csv('D:/PycharmProjects/EcgGmm/DataAnalysis/Data_of_train_and_test_VG',r'combine_all_class.csv')
    # print(df)
    # GD.partition_all(df)
    GD.SMOTE_train()



