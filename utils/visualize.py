import os
import matplotlib.pyplot as plt


from src.constants import RESULT_DIR

def save_results(train_losses, test_losses, train_accuracies, test_accuracies):
    os.makedirs(RESULT_DIR, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, test_losses, label='Test')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train')
    plt.plot(epochs, test_accuracies, label='Test')
    plt.title('Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(RESULT_DIR, 'train_test_results.png'))
    plt.close()