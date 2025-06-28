# train.py
import os
import copy
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import *
from dataset import CustomDataset
from model import VAEUNet3D, vae_loss

# Helper to build file paths
train_data_paths1 = [TRAIN_PATHS["tau"].format(i) for i in range(TRAIN_PATHS["count"])]
train_data_paths2 = [TRAIN_PATHS["dm"].format(i) for i in range(TRAIN_PATHS["count"])]
train_data_paths3 = [TRAIN_PATHS["galaxy"].format(i) for i in range(TRAIN_PATHS["count"])]

test_data_paths1 = [TEST_PATHS["tau"].format(i) for i in range(TEST_PATHS["count"])]
test_data_paths2 = [TEST_PATHS["dm"].format(i) for i in range(TEST_PATHS["count"])]
test_data_paths3 = [TEST_PATHS["galaxy"].format(i) for i in range(TEST_PATHS["count"])]

# Datasets and loaders
train_dataset = CustomDataset(train_data_paths1, train_data_paths2, train_data_paths3)
test_dataset = CustomDataset(test_data_paths1, test_data_paths2, test_data_paths3)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# Model and optimizer
model = VAEUNet3D().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Optional resume
if START_EPOCH > 0 and os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    START_EPOCH = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {START_EPOCH}")
else:
    print("Starting training from scratch.")

# Training loop
best_loss = float('inf')
history = {'train_loss': [], 'test_loss': []}
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(START_EPOCH, NUM_EPOCHS):
    model.train()
    running_train_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs, mu, logvar = model(inputs)
        loss = vae_loss(outputs, targets, mu, logvar)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)

    # Validation
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, mu, logvar = model(inputs)
            loss = vae_loss(outputs, targets, mu, logvar)
            running_test_loss += loss.item()

    epoch_test_loss = running_test_loss / len(test_loader)

    history['train_loss'].append(epoch_train_loss)
    history['test_loss'].append(epoch_test_loss)

    # Logging
    with open(LOG_PATH, 'a') as f:
        f.write(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}\n")

    # Checkpointing
    if (epoch + 1) % 50 == 0 or epoch_test_loss < best_loss:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_test_loss,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved at epoch {epoch+1}")

    if epoch_test_loss < best_loss:
        best_loss = epoch_test_loss
        best_model_wts = copy.deepcopy(model.state_dict())

# Save final model
model.load_state_dict(best_model_wts)
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history,
}, FINAL_MODEL_PATH)
print(f"Training completed. Model saved to {FINAL_MODEL_PATH}")

# Plot training curve
if SAVE_PLOT:
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['test_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(HISTORY_PLOT)
    plt.close()
    print("Training history plot saved.")

