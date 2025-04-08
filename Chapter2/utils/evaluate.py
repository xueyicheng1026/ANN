import torch
from tqdm import tqdm

def evaluate_model(model, test_loader, criterion, device):
    """验证函数"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="evaluate"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects.float() / len(test_loader.dataset)

    print(f"test Loss: {total_loss:.4f} Acc: {total_acc:.4f}")
    return total_loss, total_acc.item()