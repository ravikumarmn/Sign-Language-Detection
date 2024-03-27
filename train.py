import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# local imports
from utils.visualize import save_results
from src.model import CustomLSTM
from src.data_loader import train_loader, test_loader, train_dataset, test_dataset
from src.constants import *


# Initialize the model
model = CustomLSTM().to(DEVICE)
print(f"Model set to {DEVICE}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode=SCHEDULER_MODE, patience=5, factor=SCHEDULER_FACTOR, verbose=True)

# Train the model
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels.squeeze()).sum().item()
        train_total += labels.size(0)
    
    train_loss /= len(train_dataset)
    train_accuracy = train_correct / train_total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Testing phase
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels.squeeze()).sum().item()
            test_total += labels.size(0)

    test_loss /= len(test_dataset)
    test_accuracy = test_correct / test_total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # Update learning rate scheduler based on validation loss
    scheduler.step(test_loss)
    current_lr = scheduler.get_last_lr()[0] 

    
    # Early stopping check
    if test_loss < best_val_loss - MIN_DELTA:
        best_val_loss = test_loss
        cnt_since_last_improvement = 0
    else:
        cnt_since_last_improvement += 1
        if cnt_since_last_improvement >= PATIENCE:
            print("Early stopping! Validation loss has not improved.")
            break
        
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

torch.save(model.state_dict(), CHECKPOINT_PATH)
save_results(train_losses, test_losses, train_accuracies, test_accuracies)
print("Training completed.")