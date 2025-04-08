import torch
import torch.nn as nn
import torch.optim as optim
from jupyter_core.version import pattern
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
import re
import time
import json
import datetime

from utils.evaluate import evaluate_model

def train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        num_epochs=25,
        use_amp=True,
        scheduler=None,
        early_stop_patience=5,
        save_interval=5,
        print_freq=1,
        load_last_model=False,
        load_best_model=False,
        finetune=False,  # 添加微调标记
        from_scratch=False  # 添加从头训练标记
):
    """
    参数:
    - model: 要训练的模型
    - train_loader: 训练数据加载器
    - test_loader: 验证数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - device: 训练设备 (cuda/cpu)
    - num_epochs: 训练轮次
    - use_amp: 是否使用混合精度训练
    - scheduler: 学习率调度器
    - early_stop_patience: 早停耐心值
    - save_interval: 模型断点保存间隔
    - print_freq: 打印频率
    - load_last_model: 是否从上次训练的最后模型恢复
    - load_best_model: 是否从最佳模型恢复
    - finetune: 是否为微调模式
    - from_scratch: 是否为从头训练模式
    """

    # 初始化
    scaler = GradScaler(enabled=use_amp)
    best_test_acc = 0.0
    epochs_no_improve = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'training_time': [],  # 添加训练时间记录
        'epoch_time': [],     # 每个epoch的时间
        'total_time': 0,      # 总训练时间
        'finetune': finetune,  # 是否为微调
        'from_scratch': from_scratch  # 是否为从头训练
    }

    model_name = model.__class__.__name__
    train_mode = "finetune" if finetune else "scratch" if from_scratch else "pretrained"
    
    # 更新模型名称以包含训练模式
    model_save_name = f"{model_name}_{train_mode}"

    # 创建保存目录
    save_dir = os.path.join("save", model_save_name)
    print(f"模型将保存到: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # 为history创建单独目录
    history_dir = os.path.join("save", "history")
    os.makedirs(history_dir, exist_ok=True)

    # 初始化训练起点
    start_epoch = 0
    
    # 记录开始时间
    training_start_time = time.time()

    # 加载最近一次断点
    if load_last_model:
        checkpoint_files = []
        pattern = re.compile(rf"{model_save_name}_(\d+)\.pth")
        for file in os.listdir(save_dir):
            match = pattern.match(file)
            if match:
                epoch_num = int(match.group(1))
                checkpoint_files.append((epoch_num, file))

        if checkpoint_files:
            # 找到最大的epoch
            checkpoint_files.sort(reverse=True, key=lambda x: x[0])
            max_epoch, latest_file = checkpoint_files[0]
            last_model_path = os.path.join(save_dir, latest_file)
            checkpoint = torch.load(last_model_path)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint.get('epoch', 0)
            best_test_acc = checkpoint.get('best_test_acc', 0.0)
            history = checkpoint.get('history', history)
            print(f"恢复训练从第 {start_epoch} 轮开始，最佳验证准确率 {best_test_acc:.4f}")
        else:
            print("未找到断点文件，开始新训练")

    # 加载最佳模型
    if load_best_model:
        best_model_path = os.path.join(save_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_test_acc = checkpoint.get('best_test_acc', 0.0)
            print(f"加载最佳模型，验证准确率 {best_test_acc:.4f}")
        else:
            print("未找到最佳模型文件")

    print(f"开始训练！\nDevice: {device}, Model: {model_save_name}, Mode: {train_mode}")
    # 打印CUDA相关信息，便于检查GPU是否正常工作
    if device.type == 'cuda':
        print(f"使用GPU: {torch.cuda.get_device_name(device)}")
        if torch.cuda.is_available():
            print(f"GPU可用内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()
        print(f"\n Epoch {epoch + 1}/{start_epoch + num_epochs}")
        print("-" * 30)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 混合精度训练
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, 1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计信息
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        # 验证阶段
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # 记录本轮训练时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        history['epoch_time'].append(epoch_time)
        history['training_time'].append(time.time() - training_start_time)
        
        if (epoch + 1) % print_freq == 0:
            print(f"Epoch {epoch + 1} : train_loss: {epoch_loss:.4f}, train_acc: {epoch_acc:.4f}, "
                  f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, "
                  f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                  f"time: {epoch_time:.2f}s")

        # 学习率调度
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # 定期保存模型
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_test_acc': best_test_acc,
                'history': history
            }
            save_path = os.path.join(save_dir, f"{model_save_name}_{epoch + 1}.pth")
            torch.save(checkpoint, save_path)
            print(f"模型断点已保存至 {save_path}")

        # 更新最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_test_acc': best_test_acc,
                'history': history
            }
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(best_checkpoint, best_path)
            epochs_no_improve = 0
            print(f"新最佳模型，准确率 {test_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"早停触发，停止训练")
                break

    # 记录总训练时间
    total_training_time = time.time() - training_start_time
    history['total_time'] = total_training_time
    
    print(f"总训练时间: {datetime.timedelta(seconds=total_training_time)}")
    
    # 训练结束加载最佳模型
    best_model_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    
    # 保存训练历史记录到单独的JSON文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(history_dir, f"{model_save_name}_{timestamp}.json")
    with open(history_path, 'w') as f:
        # 确保所有数据都是JSON可序列化的
        json_history = {k: v if isinstance(v, (list, dict, str, int, float, bool)) else str(v) 
                         for k, v in history.items()}
        json.dump(json_history, f, indent=4)
    print(f"训练历史已保存至 {history_path}")

    return model, history
