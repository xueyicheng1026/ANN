import torch
import os
import torchvision.models as models
from torch import nn, optim
import json
import datetime
from utils.data_loader import load_cifar10
from utils.train import train_model
from utils.predict import predict_and_visualize
from utils.evaluate import evaluate_model

# 模型配置字典
MODEL_CONFIGS = {
    'densenet121': {
        'model_fn': models.densenet121,
        'classifier': 'classifier', 
        'feature_dim': 1024,
        'blocks': ['features.denseblock4', 'features.denseblock3']
    },
    'densenet169': {
        'model_fn': models.densenet169,
        'classifier': 'classifier', 
        'feature_dim': 1664,
        'blocks': ['features.denseblock4', 'features.denseblock3']
    },
    'resnext50': {
        'model_fn': models.resnext50_32x4d,
        'classifier': 'fc', 
        'feature_dim': 2048,
        'blocks': ['layer4', 'layer3']
    },
    'resnext101': {
        'model_fn': models.resnext101_32x8d,
        'classifier': 'fc', 
        'feature_dim': 2048,
        'blocks': ['layer4', 'layer3']
    },
    'mobilenet_v2': {
        'model_fn': models.mobilenet_v2,
        'classifier': 'classifier.1', 
        'feature_dim': 1280,
        'blocks': ['features.18', 'features.17']
    },
    'resnet18': {
        'model_fn': models.resnet18,
        'classifier': 'fc', 
        'feature_dim': 512,
        'blocks': ['layer4', 'layer3']
    },
    'shufflenet_v2': {
        'model_fn': models.shufflenet_v2_x0_5,
        'classifier': 'fc', 
        'feature_dim': 1024,
        'blocks': ['conv5', 'stage4']
    }
}

def get_model(model_name, pretrained=True, num_classes=10, adapt_first_layer=False):
    """获取指定模型"""
    config = MODEL_CONFIGS[model_name]
    
    # 当pretrained=True时，如果本地没有预训练模型，将自动从互联网下载
    print(f"正在加载{model_name}模型" + ("(预训练)" if pretrained else "(无预训练)"))
    if pretrained:
        print("如果本地没有预训练权重，将会自动从互联网下载")
    
    model = config['model_fn'](pretrained=pretrained)
    
    # 修改第一层以适应小尺寸输入
    if adapt_first_layer:
        print("修改第一层卷积以适应32x32输入尺寸")
        if 'densenet' in model_name:
            # DenseNet 第一层修改
            first_conv = model.features.conv0
            model.features.conv0 = nn.Conv2d(
                in_channels=first_conv.in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=3,  # 降低kernel大小
                stride=1,       # 减小步长
                padding=1,      # 调整padding
                bias=first_conv.bias is not None
            )
        elif 'resnet' in model_name or 'resnext' in model_name:
            # ResNet/ResNeXt 第一层修改
            first_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels=first_conv.in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=3,  # 降低kernel大小
                stride=1,       # 减小步长
                padding=1,      # 调整padding
                bias=first_conv.bias is not None
            )
            # 可选：移除或修改maxpool层
            model.maxpool = nn.Identity()  # 删除maxpool，保持特征图尺寸
        elif 'mobilenet' in model_name:
            # MobileNet 第一层修改
            first_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                in_channels=first_conv.in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=3,  # 降低kernel大小
                stride=1,       # 减小步长
                padding=1,      # 调整padding
                bias=first_conv.bias is not None
            )
        elif 'shufflenet' in model_name:
            # ShuffleNet 第一层修改
            first_conv = model.conv1[0]
            model.conv1[0] = nn.Conv2d(
                in_channels=first_conv.in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=3,  # 降低kernel大小
                stride=1,       # 减小步长
                padding=1,      # 调整padding
                bias=first_conv.bias is not None
            )
            # 移除maxpool
            model.maxpool = nn.Identity()
    
    # 修改分类器
    classifier_name = config['classifier']
    feature_dim = config['feature_dim']
    
    if '.' in classifier_name:
        parent_attr, child_attr = classifier_name.split('.')
        setattr(getattr(model, parent_attr), int(child_attr), nn.Linear(feature_dim, num_classes))
    else:
        setattr(model, classifier_name, nn.Linear(feature_dim, num_classes))
    
    return model

def _get_module_by_path(model, path):
    """通过路径获取模型的子模块"""
    if '.' in path:
        parts = path.split('.')
        module = model
        for part in parts:
            try:
                idx = int(part)
                module = module[idx]
            except (ValueError, IndexError):
                if hasattr(module, part):
                    module = getattr(module, part)
                elif isinstance(module, torch.nn.Sequential) and idx < len(module):
                    module = module[idx]
                else:
                    return None
        return module
    
    if hasattr(model, path):
        return getattr(model, path)
    
    return None

def setup_finetune_params(model, model_name, finetune_mode='last_layer'):
    """配置模型微调参数"""
    config = MODEL_CONFIGS[model_name]
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻分类器
    classifier_name = config['classifier']
    if '.' in classifier_name:
        parent, child = classifier_name.split('.')
        module = getattr(model, parent)[int(child)]
    else:
        module = getattr(model, classifier_name)
    
    for param in module.parameters():
        param.requires_grad = True
    
    # 根据微调模式解冻更多层
    if finetune_mode in ['last_block', 'last_two_blocks']:
        # 解冻最后一个块
        block = _get_module_by_path(model, config['blocks'][0])
        for param in block.parameters():
            param.requires_grad = True
    
    if finetune_mode == 'last_two_blocks':
        # 解冻倒数第二个块
        block = _get_module_by_path(model, config['blocks'][1])
        for param in block.parameters():
            param.requires_grad = True
    
    # 收集需要训练的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    
    return trainable_params

def run_experiment(config):
    """运行单个实验"""
    # 设置自定义下载路径
    os.environ['TORCH_HOME'] = './models'
    print(f"预训练模型将保存在: {os.path.abspath('./models')}")
    
    # 载入数据
    train_dataset, test_dataset, train_loader, test_loader = load_cifar10(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        train_limit=config['train_limit'],
        test_limit=config['test_limit'],
        random_seed=config['seed'],
        resize_to_224=config.get('resize_to_224', True)  # 添加调整尺寸选项
    )
    
    print(f"训练集长度: {len(train_dataset)}")
    print(f"测试集长度: {len(test_dataset)}")
    
    # 确定训练模式
    finetune = config['mode'] == 'finetune'
    from_scratch = config['mode'] == 'scratch'
    
    # 获取模型
    # 如果不调整图像尺寸，则需要适应第一层
    adapt_first_layer = not config.get('resize_to_224', False)
    model = get_model(
        config['model'], 
        pretrained=not from_scratch, 
        num_classes=10,
        adapt_first_layer=adapt_first_layer
    )
    print(f"{'从头训练' if from_scratch else '微调预训练模型' if finetune else '使用预训练模型'} {config['model']}")
    model = model.to(config['device'])
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
    # 配置优化器
    if finetune and not from_scratch:
        trainable_params = setup_finetune_params(model, config['model'], config['finetune_mode'])
        optimizer = optim.AdamW(trainable_params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # 开始训练
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=config['device'],
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=config['epochs'],
        use_amp=config['use_amp'],
        early_stop_patience=config['patience'],
        save_interval=config['save_interval'],
        print_freq=config['print_freq'],
        load_last_model=config['resume'],
        load_best_model=config['use_best'],
        finetune=finetune,
        from_scratch=from_scratch
    )
    
    # 最终评估
    final_test_loss, final_test_acc = evaluate_model(
        trained_model, test_loader, criterion, config['device']
    )
    
    # 记录实验结果
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    experiment_result = {
        "model": config['model'],
        "mode": config['mode'],
        "finetune_mode": config['finetune_mode'] if config['mode'] == 'finetune' else "N/A",
        "final_accuracy": final_test_acc,
        "training_time": history.get("total_time", 0),
        "epochs": len(history.get("train_loss", [])),
        "best_accuracy": max(history.get("test_acc", [0])),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params_count": total_params,
        "trainable_params_count": trainable_params
    }
    
    print(f"最终测试准确率: {final_test_acc:.4f}")
    
    # 可视化
    if config.get('visualize', False):
        predict_and_visualize(trained_model, test_dataset, num_samples=5)
    
    return trained_model, history, experiment_result

def main():
    # 默认配置
    config = {
        'data_dir': './data',
        'train_limit': None,
        'test_limit': None,
        'batch_size': 128,
        'model': 'densenet121',
        'mode': 'pretrained',
        'finetune_mode': 'last_layer',
        'epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'label_smoothing': 0.1,
        'patience': 10,
        'use_amp': False,
        'save_interval': 5,
        'print_freq': 1,
        'resume': False,
        'use_best': False,
        'seed': 42,
        'visualize': False,
        'resize_to_224': False,  # 是否调整输入尺寸为224×224
        'device': torch.device('cuda' if torch.cuda.is_available() else 
                              ('mps' if torch.backends.mps.is_available() else 'cpu'))
    }
    
    print(f"使用设备: {config['device']}")
    print(f"CUDA可用性: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    
    # 创建保存目录
    os.makedirs(os.path.join("save", "history"), exist_ok=True)
    os.makedirs(os.path.join("save", "results"), exist_ok=True)
    
    # 运行单个实验
    trained_model, history, result = run_experiment(config)
    print(f"实验结果: {result}")

if __name__ == "__main__":
    main()


