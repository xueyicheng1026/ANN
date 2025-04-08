import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    # 尝试多种可能的中文字体
    font_list = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'STHeiti', 'Arial Unicode MS']
    
    # 尝试找到可用的字体
    font_found = False
    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
            # 验证字体是否可用
            font_prop = FontProperties(family=font)
            if font_prop.get_name() != 'DejaVu Sans':
                font_found = True
                print(f"使用中文字体: {font}")
                break
        except:
            continue
    
    if not font_found:
        # 使用Matplotlib内置的中文支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        print("尝试使用Arial Unicode MS字体")
except:
    print("未找到中文字体，将使用默认字体")

# 设置Seaborn样式
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 定义数据目录和输出目录
DATA_DIR = "/Users/xueyicheng/Documents/人工神经网络模型与算法/Chapter2/远程训练结果/results"
OUTPUT_DIR = "/Users/xueyicheng/Documents/人工神经网络模型与算法/Chapter2/figures"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv_data(csv_file):
    """Load CSV summary data"""
    return pd.read_csv(csv_file)

def load_json_data(json_file):
    """Load JSON history data"""
    with open(json_file, 'r') as f:
        return json.load(f)

def find_history_files():
    """Find all history record files"""
    return glob.glob(os.path.join(DATA_DIR, '*_history.json'))

def find_results_table():
    """Find results summary table"""
    results_tables = glob.glob(os.path.join(DATA_DIR, 'results_table_*.csv'))
    if results_tables:
        return results_tables[0]
    return None

def plot_training_curves(history_files):
    """Plot training curves for each model"""
    for file in history_files:
        data = load_json_data(file)
        base_name = os.path.basename(file).split('_history.json')[0]
        model_name, train_mode, finetune_mode = base_name.split('_', 2)
        
        # Plot loss curve
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(data['train_loss']) + 1)
        plt.plot(epochs, data['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, data['test_loss'], 'r-', label='Validation Loss')
        plt.title(f'{model_name} - {train_mode} - {finetune_mode} Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{base_name}_loss.png'), dpi=300)
        plt.close()
        
        # Plot accuracy curve
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, [acc * 100 for acc in data['train_acc']], 'b-', label='Training Accuracy')
        plt.plot(epochs, [acc * 100 for acc in data['test_acc']], 'r-', label='Validation Accuracy')
        plt.title(f'{model_name} - {train_mode} - {finetune_mode} Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{base_name}_accuracy.png'), dpi=300)
        plt.close()

def plot_combined_accuracy_curves(history_files, results_table=None):
    """Plot validation accuracy curves comparison for all models"""
    plt.figure(figsize=(12, 8))
    
    # Colors and line types for different training modes
    colors = {
        'pretrained': 'blue',
        'finetune': 'green',
        'scratch': 'red'
    }
    
    linetypes = {
        'last_layer': ':',
        'last_block': '--',
        'last_two_blocks': '-.',
        'N/A': '-'
    }
    
    for file in history_files:
        data = load_json_data(file)
        base_name = os.path.basename(file).split('_history.json')[0]
        parts = base_name.split('_')
        
        if len(parts) >= 3:
            model_name = parts[0]
            train_mode = parts[1]
            finetune_mode = '_'.join(parts[2:]) if train_mode == 'finetune' else 'N/A'
            
            epochs = range(1, len(data['test_acc']) + 1)
            label = f'{model_name} - {train_mode}'
            if finetune_mode != 'N/A':
                label += f' ({finetune_mode})'
                
            color = colors.get(train_mode, 'black')
            linetype = linetypes.get(finetune_mode, '-')
            
            plt.plot(epochs, [acc * 100 for acc in data['test_acc']], 
                     color=color, linestyle=linetype, 
                     label=label)
    
    plt.title('Validation Accuracy Comparison of Different Models and Training Modes')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_accuracy_comparison.png'), dpi=300)
    plt.close()

def plot_bar_comparisons(results_file):
    """Plot bar charts comparing performance across different models and training modes"""
    if not results_file:
        print("Results summary table not found")
        return
    
    df = load_csv_data(results_file)
    
    # Preprocess data
    df['model_mode'] = df['model'] + '_' + df['mode']
    if 'finetune_mode' in df.columns:
        # Generate more specific labels for fine-tuning mode
        df.loc[df['mode'] == 'finetune', 'model_mode'] = df.loc[df['mode'] == 'finetune'].apply(
            lambda row: f"{row['model']}_{row['mode']}_{row['finetune_mode']}", axis=1)
    
    # Accuracy comparison
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='model_mode', y='final_accuracy', data=df, 
                     palette='viridis', hue='model')
    plt.title('Final Accuracy of Different Models and Training Modes')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model and Training Mode')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    # Add value labels on bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2%}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'), dpi=300)
    plt.close()
    
    # Training time comparison
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='model_mode', y='training_time', data=df, 
                     palette='rocket', hue='model')
    plt.title('Training Time of Different Models and Training Modes')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model and Training Mode')
    plt.ylabel('Training Time (seconds)')
    plt.tight_layout()
    # Add time labels on bars (formatted as min:sec)
    for i, p in enumerate(ax.patches):
        minutes, seconds = divmod(p.get_height(), 60)
        ax.annotate(f'{int(minutes)}:{int(seconds):02d}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_time_comparison.png'), dpi=300)
    plt.close()
    
    # Parameter efficiency comparison (accuracy/parameter count)
    if 'params_count' in df.columns and 'trainable_params_count' in df.columns:
        df['param_efficiency'] = df['final_accuracy'] / (df['trainable_params_count'] / 1000000)  # accuracy/million params
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='model_mode', y='param_efficiency', data=df, 
                         palette='mako', hue='model')
        plt.title('Parameter Efficiency of Different Models and Training Modes (Accuracy/Million Parameters)')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Model and Training Mode')
        plt.ylabel('Parameter Efficiency')
        plt.tight_layout()
        # Add efficiency values on bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points')
        plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_efficiency.png'), dpi=300)
        plt.close()

def plot_accuracy_vs_params(results_file):
    """Plot scatter chart of accuracy vs parameter count"""
    if not results_file:
        return
    
    df = load_csv_data(results_file)
    
    if 'trainable_params_count' in df.columns and 'final_accuracy' in df.columns:
        plt.figure(figsize=(10, 8))
        
        # Create different markers for different models and training modes
        markers = {'resnext50': 'o', 'densenet121': 's'}
        colors = {'pretrained': 'blue', 'finetune': 'green', 'scratch': 'red'}
        
        for model in df['model'].unique():
            for mode in df['mode'].unique():
                subset = df[(df['model'] == model) & (df['mode'] == mode)]
                if not subset.empty:
                    plt.scatter(
                        subset['trainable_params_count'] / 1000000,  # Convert to million parameters
                        subset['final_accuracy'],
                        s=100,
                        marker=markers.get(model, 'o'),
                        color=colors.get(mode, 'black'),
                        alpha=0.7,
                        label=f'{model} - {mode}'
                    )
        
        # Add data point labels
        for i, row in df.iterrows():
            label = row['model']
            if row['mode'] == 'finetune' and 'finetune_mode' in df.columns:
                label += f"\n({row['finetune_mode']})"
            plt.annotate(
                label,
                (row['trainable_params_count'] / 1000000, row['final_accuracy']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        plt.xscale('log')  # Use log scale for better display of parameter range
        plt.title('Accuracy vs Trainable Parameter Count')
        plt.xlabel('Trainable Parameter Count (millions)')
        plt.ylabel('Accuracy')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_params.png'), dpi=300)
        plt.close()

def main():
    # Find all history record files
    history_files = find_history_files()
    if not history_files:
        print("No training history record files found")
        return
    
    # Find results summary table
    results_file = find_results_table()
    
    # Plot training curves for each model
    plot_training_curves(history_files)
    
    # Plot validation accuracy comparison for all models
    plot_combined_accuracy_curves(history_files, results_file)
    
    # Plot bar chart comparisons
    if results_file:
        plot_bar_comparisons(results_file)
        
        # Plot accuracy vs parameter count scatter chart
        plot_accuracy_vs_params(results_file)
    
    print(f"All charts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
