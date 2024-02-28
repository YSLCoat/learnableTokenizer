from spatial_transformer import spatialTransformer
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import litdata.litdata as litdata
import torchvision
#from superpixelAlgorithms.sliclayer import SLICLayer
from skimage.segmentation import slic 
from skimage.util import img_as_float
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SLICLayer(nn.Module):
    def __init__(self, n_segments=100, compactness=10):
        super(SLICLayer, self).__init__()
        self.n_segments = n_segments
        self.compactness = compactness

    def forward(self, batch_images):
        segmented_batch = []
        for image in batch_images:
            segments = self.slic(image)
            segmented_batch.append(segments)
        return segmented_batch

    def slic(self, image):
        image = image.permute(1, 2, 0).cpu().numpy()
        segments = slic(image, n_segments=self.n_segments, compactness=self.compactness)
        return segments


    
class DenseClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DenseClassifier, self).__init__()
        layers = []
        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Initialize your STN and classifier
stn = spatialTransformer(n_channels=3).to(device)
# model = CNNClassifier().to(device)
model = DenseClassifier(100, [50, 100], 100).to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(stn.parameters()) + list(model.parameters()), lr=0.001)

slic_layer = SLICLayer(n_segments=10, compactness=10)


postprocess = (
    torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    nn.Identity(),
    )


train_dataset = litdata.LITDataset(
    r"IN1k",
    r"F:\data",
    override_extensions=[
        'jpg',
        'cls'
    ],
    train = True,
).map_tuple(*postprocess)

val_dataset = litdata.LITDataset(
    r"IN1k",
    r"F:\data",
    override_extensions=[
        'jpg',
        'cls'
    ],
    train = False,
).map_tuple(*postprocess)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
# Assuming you have your dataset and dataloader set up
image, _ = train_dataset[3]
#image = img_as_float(image)

for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    
    segmented_batch = slic_layer(data)
    
    batch_segment_features = []
    
    # Iterate over each image in the batch
    for image, segments in zip(segmented_batch, data):
        for segment_label in np.unique(segments):
            segment_mask = segments == segment_label
            segmented_image = np.copy(image)
            segmented_image[~segment_mask]
            segment_feature = stn(segmented_image.unsqueeze(0))  # Add batch dimension
            segment_features.append(segment_feature)
            
        segment_features = torch.stack(segment_features)
        
        # Append segment features and labels to batch lists
        batch_segment_features.append(segment_features)
    
    # Stack segment features and labels for the entire batch
    batch_segment_features = torch.cat(batch_segment_features, dim=0)
    batch_segment_labels = torch.tensor(batch_segment_labels, dtype=torch.long, device=device)
    
    assert 0, batch_segment_features.shape

   

    # Train the classifier using segment features and labels
    optimizer.zero_grad()
    outputs = model(batch_segment_features)
    loss = criterion(outputs, segment_labels)
    loss.backward()
    optimizer.step()
