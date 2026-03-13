
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from model import ECG_LSTM, ECG_BiLSTM, AMHBilstmModel
from sklearn.metrics import classification_report, recall_score, f1_score, confusion_matrix
from DataLoad import load_ecg_data
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch

"""Picture format"""
FontSize: int = 16  # label font size
plt.rcParams['font.size'] = FontSize-2
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.dpi"] = 900
plt.rcParams["savefig.bbox"] = 'tight'




def save_log(message, file_name):
    with open(file_name, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    print("\n===================== 开始训练 =====================")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.to(device)

    best_val_f1 = 0.0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )


    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(epochs):
        model.train()
        total = 0.0
        correct = 0.0
        running_loss = 0.0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            # 修改这里：只取第一个输出（logits）
            outputs, _ = model(features)  # 解包元组，只取logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()


        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # 同样修改验证函数调用
        val_acc, val_loss, val_true, val_pred, val_pre_probs = test_model(model, val_loader, device)
        val_recall = recall_score(val_true, val_pred, average='macro')
        val_f1 = f1_score(val_true, val_pred, average='macro')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'Epoch {epoch + 1}: ')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc: .4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc: .4f}')
        print(classification_report(val_true, val_pred, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4))
        save_log(f'Epoch {epoch + 1}: ', 'log.txt')
        message = str(classification_report(val_true, val_pred, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4))
        save_log(message, 'log.txt')

        scheduler.step(val_loss)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                model, epoch + 1, optimizer, train_acc, val_acc, val_f1, val_recall, 'AMH_BiLSTM'
            )


    # 训练结束后绘制指标曲线
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.title('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('checkpoints/training_metrics_AMH_BiLSTM_128_SMOTE.png')   # 这里也应该修改
    plt.close()
    print('Finished Training')

def save_checkpoint(model, epochs, optimizer, train_acc, val_acc, recall, f1, net_name):
    checkpoint = {
        'model': model.state_dict(),
        'epochs': epochs,
        'optimizer': optimizer.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'f1': f1,
        'recall': recall,
    }
    torch.save(checkpoint, f'checkpoints/best_ecg_{net_name}(128,0.001,1)_SMOTE_f1.pth')


def test_model(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true = []
    all_pred = []
    all_pred_probs = []
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    with torch.no_grad():  # 测试阶段禁用梯度计算
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # ========== 关键修复：处理模型返回的元组 ==========
            model_output = model(inputs)
            # 如果模型返回的是 (output, attn_weights, ...)，取第一个元素作为预测得分
            outputs = model_output[0] if isinstance(model_output, tuple) else model_output

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_pred_probs.extend(probs.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predicted.cpu().numpy())

    # 计算准确率和平均损失
    acc = correct / total
    avg_loss = total_loss / len(test_loader)
    # 转换预测概率为Numpy数组（适配ROC函数）
    all_pred_probs = np.array(all_pred_probs)

    return acc, avg_loss, all_true, all_pred, all_pred_probs


def test_main(test_loader, device):
    print("\n===================== 开始测试 =====================")

    def load_trained_model(model_path, device):
        """加载训练好的模型"""
        model = AMHBilstmModel(
            num_features=16,
            num_classes=5,
            cnn_channels=64,
            lstm_hidden=128,
            lstm_layers=1,
            num_heads=4,
            dropout=0.3,
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        return model

    model = load_trained_model('checkpoints/best_ecg_AMH_BiLSTM(128,0.001,1)_SMOTE_f1.pth', device)

    # 注意：需要修改 test_model 使其返回预测概率（而非仅预测标签）
    acc, loss, all_true, all_pred, all_pred_probs = test_model(model, test_loader, device)

    print('Acc: {:.4f}'.format(acc))
    print(classification_report(all_true, all_pred, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4))
    # compute_confusion_matrix(all_true, all_pred)

    # 调用ROC曲线绘制函数
    draw_ROC(all_true, all_pred_probs, n_classes=5)

def compute_confusion_matrix(all_true, all_pred):
    class_names = ['N', 'S', 'V', 'F', 'Q']
    cnf_matrix = confusion_matrix(all_true, all_pred)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    # plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig('results/confusion_matrix_BiLSTM_skip_50_128.png', dpi=1000)

def draw_ROC(all_true, all_pred_probs, n_classes=5):
    """
    绘制多分类任务的ROC曲线（One-vs-Rest 方式）
    参数:
        all_true: 真实标签列表/数组 (形状: [n_samples])
        all_pred_probs: 模型对每个类别的预测概率 (形状: [n_samples, n_classes])
        n_classes: 类别数量，默认5（对应N/S/V/F/Q）
    """
    # 1. 将真实标签二值化（One-vs-Rest）
    y_true = label_binarize(all_true, classes=[0, 1, 2, 3, 4])  # 假设类别编码为0-4对应N/S/V/F/Q

    # 2. 检查输入形状
    if y_true.shape[1] != n_classes:
        raise ValueError(f"真实标签二值化后维度为 {y_true.shape[1]}，与类别数 {n_classes} 不匹配！")
    if all_pred_probs.shape[1] != n_classes:
        raise ValueError(f"预测概率维度为 {all_pred_probs.shape[1]}，与类别数 {n_classes} 不匹配！")

    # 3. 计算每个类别的ROC曲线和AUC值
    fpr = dict()  # 假阳性率
    tpr = dict()  # 真阳性率
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], all_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 4. 计算微平均ROC曲线和AUC（可选，体现整体性能）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), all_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 5. 绘制ROC曲线
    plt.figure(figsize=(8, 7))
    class_names = ['N', 'S', 'V', 'F', 'Q']  # 对应0-4类
    # colors = ['blue', 'red', 'green', 'orange', 'purple']  # 每个类别对应颜色
    # colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    colors = ['#547BB4', '#C0321A', '#629C35', '#DD7C4F', '#6C61AF']
    # 绘制每个类别的ROC曲线
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.4f})'
        )

    # 绘制微平均ROC曲线（可选）
    plt.plot(
        fpr["micro"], tpr["micro"], color='black', linestyle='--', lw=2,
        label=f'Micro-average(AUC = {roc_auc["micro"]:.4f})'
    )

    # 绘制随机猜测的参考线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier', color='#6F6F6F')

    # 图表样式设置
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    # plt.grid(alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    # 保存图片（建议与混淆矩阵同目录）
    plt.savefig('results/ROC_curve_BiLSTM_skip_50_128.png', dpi=1000, bbox_inches='tight')
    # plt.show()




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = load_ecg_data(batch_size=32)
    # 初始化BiLSTM模型

    model = AMHBilstmModel(
        num_features=16,
        num_classes=5,
        cnn_channels=64,
        lstm_hidden=128,
        lstm_layers=1,
        num_heads=4,
        dropout=0.3,
    ).to(device)

    # train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001)
    test_main(test_loader, device)


