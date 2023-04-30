import matplotlib.pyplot as plt


def evaluate_model(history):
    # Check how loss & mae went down
    epoch_loss = history.history['loss']
    epoch_val_loss = history.history['val_loss']
    epoch_mae = history.history['mae']
    epoch_val_mae = history.history['val_mae']

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(0, len(epoch_loss)), epoch_loss, 'b-', linewidth=2, label='Train Loss')
    plt.plot(range(0, len(epoch_val_loss)), epoch_val_loss, 'r-', linewidth=2, label='Val Loss')
    plt.title('Evolution of loss on train & validation datasets over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.subplot(1,2,2)
    plt.plot(range(0,len(epoch_mae)), epoch_mae, 'b-', linewidth=2, label='Train MAE')
    plt.plot(range(0,len(epoch_val_mae)), epoch_val_mae, 'r-', linewidth=2,label='Val MAE')
    plt.title('Evolution of MAE on train & validation datasets over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend(loc='best')

    plt.savefig('../reports/error_evaluations_on_epochs.png')

    plt.show()