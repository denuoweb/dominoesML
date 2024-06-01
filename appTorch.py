import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
from PIL import Image

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, outputs, targets):
        """
        Custom loss function that penalizes prediction errors.

        Args:
        - outputs: Predicted values from the model.
        - targets: Actual values (ground truth).

        Returns:
        - loss: Computed loss value.
        """
        # Ensure outputs and targets are float tensors
        outputs = outputs.float()
        targets = targets.float()
        
        # Compute the absolute difference
        abs_diff = torch.abs(outputs - targets)
        
        # Implement a custom penalty function; this is just an example
        # You might want to design this based on your specific needs
        # For instance, penalizing underestimates more than overestimates or vice versa
        penalty = torch.where(outputs < targets, abs_diff * 2, abs_diff)  # Example: double penalty for underestimation
        
        # Compute the mean loss
        loss = torch.mean(penalty)
        
        return loss

# Define the dataset class for domino dots
class DominoDotsDataset(Dataset):
    # Constructor for the dataset class
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        
        # Adjust label extraction to handle both '69.jpg' and '60_2.jpg' formats
        self.labels = []
        for filename in os.listdir(directory):
            # Split filename at the underscore or dot
            parts = filename.split('_') if '_' in filename else filename.split('.')
            try:
                # Attempt to convert the first part to an integer
                label = int(parts[0])
            except ValueError:
                # Handle the case where conversion fails
                print(f"Warning: Filename '{filename}' does not start with an integer. Skipping.")
                continue  # Skip this file
            self.labels.append(label)
            
        print(f"Dataset initialized with {len(self.image_paths)} images")
        
    # Return the length of the dataset
    def __len__(self):
        return len(self.image_paths)
    
    # Fetch a single item by index
    def __getitem__(self, idx):
        # Get the image path and load the image using PIL
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = self.labels[idx]
        
        # Apply the transform if it's defined
        if self.transform:
            image = self.transform(image)
        
        # Return the transformed image and its corresponding label
        return image, label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define your transforms to be applied on images
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    # Randomly flip images horizontally to augment the dataset
    transforms.RandomHorizontalFlip(),
    # Randomly rotate images by 20 degrees to augment the dataset
    transforms.RandomRotation(20),
    # Convert images to PyTorch tensors
    transforms.ToTensor(),
    # Normalize images with mean and standard deviation from ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Path to the directory where the training images are located
dataset_directory = 'basePhotos/train'  # replace with the path to your dataset
print(f"Dataset directory set to: {dataset_directory}")
val_directory = 'basePhotos/validate'
test_directory = 'basePhotos/test'




# Create an instance of the DominoDotsDataset
domino_dataset = DominoDotsDataset(directory=dataset_directory, transform=transform)
val_dataset = DominoDotsDataset(directory=val_directory, transform=transform)
test_dataset = DominoDotsDataset(directory=test_directory, transform=transform)

# Create a DataLoader to load images in batches
batch_size = 7  # The size of the batch to process images in groups
valbatch_size = 1

print(f"DataLoader will use batch size of {batch_size}")
dataloader = DataLoader(domino_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=valbatch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=valbatch_size, shuffle=False)

# Load a pre-trained ResNet18 model from torchvision
model = models.resnet18(pretrained=True)
print("Pre-trained ResNet18 model loaded")

# Freeze all the layers in the pre-trained model to avoid updating their weights during training
for param in model.parameters():
    param.requires_grad = False
print("All model layers have been frozen")

for param in model.layer4.parameters():
    param.requires_grad = True
print("All layer 4 has been unfrozen")


# Modify the last layer of the model for regression task (predicting a single continuous variable)
num_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.BatchNorm1d(num_features),
    torch.nn.Linear(num_features, 1)
    )  # Output one neuron since we are predicting a count
print("Replaced the model's last layer for regression task")

# Move the model to the specified device (GPU or CPU)
model.to(device)
print("Model moved to the device")

# Define the optimizer that will update the weights of the last layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=.01, weight_decay=1e-5)
print("Optimizer defined")

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
print("Scheduler defined")

# Instantiate the custom loss function
custom_loss_function = CustomLoss()

# Define the Mean Squared Error Loss function for regression
criterion = torch.nn.L1Loss()
print("Loss function defined for regression")

# Training loop - this is where the model learns
num_epochs = 100  # Number of passes through the entire dataset
print(f"Starting training for {num_epochs} epochs")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} started")
    for i, (images, labels) in enumerate(dataloader):
        print(f"Processing batch {i+1}/{len(dataloader)}")
        
        # Move the data to the device (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device).float()  # Ensure labels are float for regression

        # Forward pass
        outputs = model(images).squeeze()  # Use squeeze() to remove singleton dimensions

        # Check and adjust the shape of labels if necessary
        if outputs.dim() < labels.dim():
            labels = labels.squeeze()  # Squeeze labels to match output dimension
        
        # Compute loss
        loss = criterion(outputs, labels)
        #customLoss = custom_loss_function(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every few batches
        if (i + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch {i+1}, Loss: {loss.item():.4f}]')

    # Validation phase
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        val_loss = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images).squeeze()
            # Match output and label dimensions
            if outputs.dim() < labels.dim():
                labels = labels.squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            scheduler.step(val_loss)
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    best_val_loss = val_loss

    # Check if the current validation loss is the best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the model
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}')

# Testing phase after training is complete
model.eval()  # Ensure model is in evaluation mode
with torch.no_grad():
    test_loss = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')