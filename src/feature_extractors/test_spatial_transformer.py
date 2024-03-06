from spatial_transformer import spatialTransformer, AttentionSpatialTransformer
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
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



#stn = spatialTransformer(n_channels=3).to(device)
stn = AttentionSpatialTransformer(n_channels=3).to(device)
# model = CNNClassifier().to(device)
model = DenseClassifier(21168, [50, 100], 1000).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(stn.parameters()) + list(model.parameters()), lr=0.001)

slic_layer = SLICLayer(n_segments=10, compactness=1000)


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
image, _ = train_dataset[3]
#image = img_as_float(image)

train_loss, train_acc = 0, 0
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        segmented_batch = slic_layer(data)
        
        batch_segment_features = []
        
        # Iterate over each image in the batch
        for image, segments in zip(data, segmented_batch):
            segmented_features = []
            for segment_label in np.unique(segments):
                segment_mask = segments == segment_label
                segmented_image = image.clone()
                segmented_image[:, ~segment_mask]
                segmented_feature = stn(segmented_image.unsqueeze(0))  # Add batch dimension
                segmented_features.append(segmented_feature)
                
            batch_segment_features.append(segmented_features)
        
        batch_segment_features_tensor = [torch.stack(segmented_features) for segmented_features in batch_segment_features]
        batch_segment_features_flattened = [segmented_features.view(segmented_features.size(0), -1) for segmented_features in batch_segment_features_tensor]
        
        max_num_segments = max(len(segmented_features) for segmented_features in batch_segment_features_flattened)

        # Pad or truncate the segmented features to ensure they all have the same length
        for i in range(len(batch_segment_features_flattened)):
            num_segments = batch_segment_features_flattened[i].size(0)
            if num_segments < max_num_segments:
                # If the number of segments is less than the maximum, pad with zeros
                padding_size = max_num_segments - num_segments
                padding = torch.zeros(padding_size, batch_segment_features_flattened[i].size(1), device=device)
                batch_segment_features_flattened[i] = torch.cat([batch_segment_features_flattened[i], padding], dim=0)
            elif num_segments > max_num_segments:
                # If the number of segments is greater than the maximum, truncate
                batch_segment_features_flattened[i] = batch_segment_features_flattened[i][:max_num_segments]

        batch_segment_features_flattened = [segmented_features.view(-1) for segmented_features in batch_segment_features_flattened]
        batch_segment_features_flattened = torch.stack(batch_segment_features_flattened)
        
        #print(batch_segment_features_flattened.shape)
    
        optimizer.zero_grad()
        outputs = model(batch_segment_features_flattened)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        correct = (y_pred_class == target).float().sum()  # Count number of correct predictions
        
        print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {correct:.7f}")

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)
    
    print(f"Epoch {epoch + 1} train loss: {train_loss:.4f}, train accuracy: {train_acc:.4f}")
