import numpy as np
import matplotlib.pyplot as plt

# Load the data
training_loss = np.load('T_loss.npy')  # Training loss file
validation_loss = np.load('V_loss.npy')  # Validation loss file
# Define the number of epochs to plot
epochs = 23
x = np.arange(1, epochs + 1)  # Epoch numbers

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, training_loss[:epochs], label="Training Loss", marker='o')
plt.plot(x, validation_loss[:epochs], label="Validation Loss", marker='s')

# Labels and title
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss over 20 Epochs")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Loss.jpg')