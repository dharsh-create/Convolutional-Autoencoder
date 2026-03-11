# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset


## DESIGN STEPS

### STEP 1:
Import necessary libraries including PyTorch, torchvision, and matplotlib.

### STEP 2:
Load the MNIST dataset with transforms to convert images to tensors.

### STEP 3:
Add Gaussian noise to training and testing images using a custom function.

### STEP 4:
Initialize model, define loss function (MSE) and optimizer (Adam).

### STEP 5:
Train the model using noisy images as input and original images as target.

### STEP 6:
Visualize and compare original, noisy, and denoised images.



## PROGRAM
### Name: DHARSHINI V
### Register Number: 212223040038
```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12)  # Latent dimension
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Output pixels are between 0 and 1
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the image
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 28, 28) # Reshape back to image
        return x
def train(model, loader, criterion, optimizer, epochs=5):
    model.train() # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(device)
            noisy_data = add_noise(data) # Add noise to the input images

            optimizer.zero_grad()
            outputs = model(noisy_data)
            loss = criterion(outputs, data) # Compare output with original data
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0: # Print loss every 100 batches
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')
        print(f'Epoch {epoch+1} finished, Average Loss: {running_loss/len(loader):.4f}')
        
```

## OUTPUT

### Model Summary

<img width="687" height="502" alt="Screenshot (1)" src="https://github.com/user-attachments/assets/5e836dfd-0c8e-4189-a6e3-4816608862cc" />


### Original vs Noisy Vs Reconstructed Image
<img width="1044" height="557" alt="Screenshot (3)" src="https://github.com/user-attachments/assets/c23fbb3f-89a2-40b9-806c-c5393535f98b" />


## RESULT
The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
