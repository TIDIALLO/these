"""
Expérience: Comparaison des activations ReLU vs GELU vs SiLU
sur CIFAR-10 avec un petit CNN

Objectif: Établir une baseline et comprendre les différences
de comportement entre les activations
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 0.001,
    'activations': ['relu', 'gelu', 'silu'],
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': Path('../../experiments/cifar10/activation_comparison'),
}


# =============================================================================
# Modèle
# =============================================================================

class SmallCNN(nn.Module):
    """
    Petit CNN pour CIFAR-10
    3 couches conv + 2 couches FC
    """

    def __init__(self, activation_type='relu'):
        super().__init__()

        # Sélection de l'activation
        activations = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
        }
        Act = activations[activation_type]

        self.features = nn.Sequential(
            # Conv block 1: 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            Act(),
            nn.MaxPool2d(2),

            # Conv block 2: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            Act(),
            nn.MaxPool2d(2),

            # Conv block 3: 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            Act(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            Act(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

        self.activation_type = activation_type

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# Data
# =============================================================================

def get_dataloaders(batch_size):
    """Charge CIFAR-10 avec augmentation standard"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root='../../resources/datasets',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root='../../resources/datasets',
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    return train_loader, test_loader


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Une époque d'entraînement"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc='Training', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Évaluation sur le test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(test_loader), 100. * correct / total


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment():
    """Exécute l'expérience complète"""

    # Setup
    torch.manual_seed(CONFIG['seed'])
    device = CONFIG['device']
    print(f"Using device: {device}")

    # Create save directory
    save_dir = CONFIG['save_dir']
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, test_loader = get_dataloaders(CONFIG['batch_size'])

    # Results storage
    results = {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'activations': {}
    }

    # Train for each activation
    for activation in CONFIG['activations']:
        print(f"\n{'='*50}")
        print(f"Training with {activation.upper()} activation")
        print('='*50)

        # Model
        model = SmallCNN(activation_type=activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG['epochs']
        )

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
        }

        # Training loop
        for epoch in range(CONFIG['epochs']):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            print(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save model
        model_path = save_dir / f"model_{activation}.pt"
        torch.save(model.state_dict(), model_path)

        # Store results
        results['activations'][activation] = {
            'history': history,
            'final_test_acc': history['test_acc'][-1],
            'best_test_acc': max(history['test_acc']),
        }

    # Save results
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Plot results
    plot_results(results, save_dir)

    return results


def plot_results(results, save_dir):
    """Génère les graphiques de comparaison"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'relu': 'blue', 'gelu': 'green', 'silu': 'orange'}

    # Loss curves
    ax = axes[0]
    for activation, data in results['activations'].items():
        epochs = range(1, len(data['history']['train_loss']) + 1)
        ax.plot(epochs, data['history']['train_loss'], '-',
                color=colors[activation], label=f'{activation} train')
        ax.plot(epochs, data['history']['test_loss'], '--',
                color=colors[activation], label=f'{activation} test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[1]
    for activation, data in results['activations'].items():
        epochs = range(1, len(data['history']['test_acc']) + 1)
        ax.plot(epochs, data['history']['test_acc'], '-',
                color=colors[activation], label=f'{activation}',
                linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'comparison_plot.png', dpi=150)
    plt.close()

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    activations = list(results['activations'].keys())
    final_accs = [results['activations'][a]['final_test_acc'] for a in activations]
    best_accs = [results['activations'][a]['best_test_acc'] for a in activations]

    x = range(len(activations))
    width = 0.35

    ax.bar([i - width/2 for i in x], final_accs, width, label='Final', color='steelblue')
    ax.bar([i + width/2 for i in x], best_accs, width, label='Best', color='coral')

    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Final vs Best Test Accuracy by Activation')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in activations])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_summary.png', dpi=150)
    plt.close()

    print(f"\nPlots saved to {save_dir}")


if __name__ == "__main__":
    results = run_experiment()

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for activation, data in results['activations'].items():
        print(f"{activation.upper():>6}: Final Acc = {data['final_test_acc']:.2f}%, "
              f"Best Acc = {data['best_test_acc']:.2f}%")
