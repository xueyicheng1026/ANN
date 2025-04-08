import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import seaborn as sns

# CIFAR10类别标签
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def show_image(dataset):
    # 随机选择一个样本
    idx = random.randint(0, len(dataset) - 1)
    image, label = dataset[idx]

    # 转换图像用于显示（反归一化）
    image_show = image.numpy().transpose((1, 2, 0))  # C,H,W -> H,W,C
    image_show = image_show * 0.2 + 0.485 # 反归一化
    image_show = np.clip(image_show, 0, 1)  # 裁剪到合理范围

    # 显示图像和真实标签
    plt.imshow(image_show)
    plt.title(f'True Label: {classes[label]}')
    plt.axis('off')
    plt.show()

    return image


def predict(model, dataset):
    # 设置模型为评估模式
    model.eval()

    image = show_image(dataset)

    # 获取设备并将数据移至对应设备
    device = next(model.parameters()).device
    image = image.unsqueeze(0).to(device)  # 增加batch维度

    # 预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    print(f'模型预测标签: {classes[predicted.item()]}')

def predict_and_visualize(model, dataset, num_samples=5):
    """预测多个样本并可视化结果"""
    # 设置模型为评估模式
    model.eval()
    
    # 随机选择样本
    indices = random.sample(range(len(dataset)), num_samples)
    images = []
    labels = []
    predictions = []
    
    # 获取设备
    device = next(model.parameters()).device
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        # 获取样本
        image, label = dataset[idx]
        labels.append(label)
        
        # 显示图像
        image_show = image.numpy().transpose((1, 2, 0))
        image_show = image_show * 0.2 + 0.485  # 反归一化
        image_show = np.clip(image_show, 0, 1)
        
        # 预测
        with torch.no_grad():
            input_tensor = image.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
        predictions.append(predicted.item())
        
        # 显示图像和预测结果
        axes[i].imshow(image_show)
        color = 'green' if predicted.item() == label else 'red'
        axes[i].set_title(f'True: {classes[label]}\nPred: {classes[predicted.item()]}', 
                         color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return labels, predictions

def visualize_training_history(history: Dict, title: str = "训练历史"):
    """可视化训练历史"""
    epochs = len(history['train_loss'])
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), history['train_loss'], 'b-', label='训练损失')
    plt.plot(range(1, epochs+1), history['test_loss'], 'r-', label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), history['train_acc'], 'b-', label='训练准确率')
    plt.plot(range(1, epochs+1), history['test_acc'], 'r-', label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def compare_models(histories: Dict[str, Dict], metric: str = 'test_acc'):
    """比较不同模型的性能"""
    plt.figure(figsize=(10, 6))
    
    for model_name, history in histories.items():
        epochs = len(history[metric])
        plt.plot(range(1, epochs+1), history[metric], '-', label=model_name)
    
    metric_name = '验证准确率' if metric == 'test_acc' else '验证损失'
    plt.title(f'模型{metric_name}对比')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, test_loader, device, title='混淆矩阵'):
    """绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for p, t in zip(all_preds, all_labels):
        conf_matrix[t, p] += 1
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.show()
    
    # 计算每个类的精确率和召回率
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    
    return precision, recall

