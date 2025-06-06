
# Importar librerías básicas para manejar arrays y gráficos
import numpy as np
import matplotlib.pyplot as plt    
import torch                      # Motor principal de PyTorch
import torch.nn as nn             # Para definir redes neuronales
import torch.optim as optim       # Optimizadores
from torchvision import datasets, transforms  # Para MNIST y transformaciones
from torch.utils.data import DataLoader       # Carga eficiente de datos

# Transformaciones: convertir a tensor y normalizar a [0,1]
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))  # media y std para MNIST
])

# Cargar conjuntos de entrenamiento y prueba
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoaders para manejar por lotes
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Capa convolucional: 1 canal entrada, 32 filtros, tamaño 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # Capa de pooling 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Capa densa intermedia: flatten primero, luego 128 neuronas
        self.fc1 = nn.Linear(32 * 13 * 13, 128)  # 28x28 -> 26x26 (conv) -> 13x13 (pool)
        # Capa de salida: 10 clases
        self.fc2 = nn.Linear(128, 10)
        # Activación ReLU
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Paso hacia adelante
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 13 * 13)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Inicializar listas para métricas
train_losses = []
train_accuracies = []

# Entrenar por 5 épocas
for epoch in range(5):
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # Resetear gradientes
        outputs = model(images)        # Forward
        loss = criterion(outputs, labels)  # Cálculo del error
        loss.backward()                # Backpropagation
        optimizer.step()               # Actualización de pesos
        
        running_loss += loss.item()

    # === Calcular pérdida promedio de la época ===
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # === Calcular precisión en entrenamiento al final de la época ===
    model.eval()  # Modo evaluación para métricas
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_accuracy = 100 * correct / total
    train_accuracies.append(epoch_accuracy)

    # Mostrar resultados de la época
    print(f"Época {epoch+1}, Pérdida promedio: {epoch_loss:.4f}, Precisión entrenamiento: {epoch_accuracy:.2f}%")

    model.train()  # Volver a modo entrenamiento para la siguiente época

plt.figure(figsize=(12, 5))

# Gráfica de pérdida por época (usando train_losses)
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Pérdida', color = 'purple')
plt.title("Pérdida de entrenamiento por época")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()

# Gráfica de precisión por época (usando train_accuracies)
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Precisión (%)', color='pink')
plt.title("Precisión de entrenamiento por época")
plt.xlabel("Época")
plt.ylabel("Precisión (%)")
plt.legend()

plt.tight_layout()
plt.show()

print(model)  # Muestra toda la arquitectura definida
