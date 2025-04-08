import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
from main import run_experiment

def run_all_experiments():
    """运行所有的模型和模式组合"""
    # 配置参数
    config = {
        'data_dir': './data',
        'train_limit': None, 
        'test_limit': None,
        'batch_size': 128,
        'epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 10,
        'use_amp': False,
        'label_smoothing': 0.1,
        'finetune_mode': 'last_layer',
        'save_interval': 5,
        'print_freq': 1,
        'resume': False,
        'use_best': False,
        'seed': 42,
        'output_dir': './results',
        'debug': True,
        'quick_test': False,
        'visualize': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 
                              ('mps' if torch.backends.mps.is_available() else 'cpu'))
    }
    
    print(f"使用设备: {config['device']}")
    print(f"CUDA可用性: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    
    # 模型和训练模式配置
    models = ['resnext50','densenet121']
    modes = ['pretrained', 'finetune', 'scratch']
    finetune_modes = ['last_layer', 'last_block', 'last_two_blocks']
    
    # 快速测试模式
    if config['quick_test']:
        config['train_limit'] = 10
        config['test_limit'] = 50
        config['epochs'] = 2
        models = ['resnet18']
        modes = ['finetune']
        finetune_modes = ['last_layer']

    # 调试模式
    if config['debug']:
        config['train_limit'] = 100
        config['test_limit'] = 500
        config['epochs'] = 2
        models = ['resnet18', 'shufflenet_v2']

    # 创建保存目录
    os.makedirs(config['output_dir'], exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = []
    
    # 运行所有实验
    for model_name in models:
        for mode in modes:
            # 对于微调模式，尝试不同的微调方式
            if mode == 'finetune':
                for ft_mode in finetune_modes:
                    print(f"\n{'='*50}")
                    print(f"开始实验: 模型={model_name}, 模式={mode}, 微调方式={ft_mode}")
                    print(f"{'='*50}\n")
                    
                    config['model'] = model_name
                    config['mode'] = mode
                    config['finetune_mode'] = ft_mode
                    
                    try:
                        trained_model, history, result = run_experiment(config)
                        
                        result["finetune_mode"] = ft_mode
                        result["params_count"] = sum(p.numel() for p in trained_model.parameters())
                        result["trainable_params_count"] = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
                        
                        results.append(result)
                        
                        # 保存历史记录
                        exp_name = f"{model_name}_{mode}_{ft_mode}"
                        history_file = os.path.join(config['output_dir'], f"{exp_name}_{timestamp}_history.json")
                        
                        with open(history_file, 'w') as f:
                            json_history = {k: v if isinstance(v, (list, dict, str, int, float, bool)) else str(v) 
                                          for k, v in history.items()}
                            json.dump(json_history, f, indent=4)
                    except Exception as e:
                        print(f"实验失败: {model_name}_{mode}_{ft_mode}, 错误: {str(e)}")
                        continue
            else:
                # 对于非微调模式，正常运行
                print(f"\n{'='*50}")
                print(f"开始实验: 模型={model_name}, 模式={mode}")
                print(f"{'='*50}\n")
                
                config['model'] = model_name
                config['mode'] = mode
                
                try:
                    trained_model, history, result = run_experiment(config)
                    
                    result["finetune_mode"] = "N/A"
                    result["params_count"] = sum(p.numel() for p in trained_model.parameters())
                    result["trainable_params_count"] = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
                    
                    results.append(result)
                    
                    # 保存历史记录
                    exp_name = f"{model_name}_{mode}"
                    history_file = os.path.join(config['output_dir'], f"{exp_name}_{timestamp}_history.json")
                    
                    with open(history_file, 'w') as f:
                        json_history = {k: v if isinstance(v, (list, dict, str, int, float, bool)) else str(v) 
                                      for k, v in history.items()}
                        json.dump(json_history, f, indent=4)
                except Exception as e:
                    print(f"实验失败: {model_name}_{mode}, 错误: {str(e)}")
                    continue
    
    # 保存结果
    results_file = os.path.join(config['output_dir'], f"all_experiments_{timestamp}.json")
    with open(results_file, 'w') as f:
        json_results = []
        for result in results:
            json_result = {}
            for k, v in result.items():
                if isinstance(v, (str, int, float, bool, list, dict)) and not (isinstance(v, float) and pd.isna(v)):
                    json_result[k] = v
                elif pd.isna(v):
                    json_result[k] = None
                else:
                    json_result[k] = str(v)
            json_results.append(json_result)
        
        json.dump(json_results, f, indent=4)
    
    # 生成结果表格
    generate_results_table(results, os.path.join(config['output_dir'], f"results_table_{timestamp}.csv"))
    
    return results

def generate_results_table(results, output_file):
    """生成实验结果表格"""
    df = pd.DataFrame(results)
    
    if 'finetune_mode' in df.columns:
        df['training_mode'] = df.apply(
            lambda row: f"{row['mode']}_{row['finetune_mode']}" if row['mode'] == 'finetune' else row['mode'], 
            axis=1
        )
    
    # 添加训练参数大小（以MB为单位）
    if 'params_count' in df.columns:
        df['params_size_mb'] = df['params_count'] * 4 / (1024 * 1024)  # 假设每个参数占用4字节
    
    # 添加可训练参数大小（以MB为单位）
    if 'trainable_params_count' in df.columns:
        df['trainable_params_mb'] = df['trainable_params_count'] * 4 / (1024 * 1024)
    
    # 格式化训练时间为易读格式
    if 'training_time' in df.columns:
        df['training_time_formatted'] = df['training_time'].apply(
            lambda x: str(datetime.timedelta(seconds=int(x))) if pd.notna(x) else "N/A"
        )
    
    # 保存为CSV
    df.to_csv(output_file, index=False)
    print(f"结果表格已保存至 {output_file}")
    
    # 输出结果摘要，增加训练时长和参数大小
    cols = ['model', 'training_mode' if 'training_mode' in df.columns else 'mode', 
            'final_accuracy', 'best_accuracy', 'training_time_formatted', 
            'params_size_mb', 'trainable_params_mb']
    
    available_cols = [col for col in cols if col in df.columns]
    print("\n实验结果汇总:")
    
    # 格式化表格输出，使结果更易读
    pd.set_option('display.float_format', '{:.2f}'.format)
    summary_df = df[available_cols].copy()
    
    if 'params_size_mb' in summary_df.columns:
        summary_df['params_size_mb'] = summary_df['params_size_mb'].apply(lambda x: f"{x:.2f} MB")
    
    if 'trainable_params_mb' in summary_df.columns:
        summary_df['trainable_params_mb'] = summary_df['trainable_params_mb'].apply(lambda x: f"{x:.2f} MB")
    
    if 'final_accuracy' in summary_df.columns:
        summary_df['final_accuracy'] = summary_df['final_accuracy'].apply(lambda x: f"{x:.4f}")
    
    if 'best_accuracy' in summary_df.columns:
        summary_df['best_accuracy'] = summary_df['best_accuracy'].apply(lambda x: f"{x:.4f}")
    
    print(summary_df)
    
    # 添加一个详细的CSV，包含所有信息
    detailed_output_file = output_file.replace('.csv', '_detailed.csv')
    df.to_csv(detailed_output_file, index=False)
    print(f"详细结果表格已保存至 {detailed_output_file}")

if __name__ == "__main__":
    results = run_all_experiments()
