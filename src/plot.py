import matplotlib.pyplot as plt

def plot_history(train_loss, val_loss, train_acc, val_acc):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))

    ax1.plot(train_loss, label="Train Loss")
    ax1.plot(val_loss, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Over Training Process")
    ax1.legend()

    ax2.plot(train_acc, label="Train Accuracy")
    ax2.plot(val_acc, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Over Training Process")
    ax2.legend()

    plt.tight_layout()
    plt.show()