import torch
from PIL import Image
from torch.optim import Adam
from torchvision import transforms

from DataSet import YOLODataset
from YoloLoss import YOLOLoss
from YoloModel import YOLOv3
from torch.utils.data import DataLoader

# Define data transforms
def transform(image, boxes):
    # Resize Image and Convert to tensor
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = transforms.Compose([
        transforms.Resize((416, 416)),  # Resize the image
        transforms.ToTensor(),  # Convert image to tensor
    ])(image)

    return image, boxes

# define collate_fn
def collate_fn(batch):
    """
        Custom collate function for the YOLO dataset.

        Args:
        - batch (list of tuples): A list where each element is a tuple (image, (bbox_targets, obj_targets, class_targets))

        Returns:
        - images (torch.Tensor): Batch of images.
        - bbox_targets (torch.Tensor): Concatenated bounding boxes for the batch.
        - obj_targets (torch.Tensor): Concatenated objectness scores for the batch.
        - class_targets (torch.Tensor): Concatenated class labels for the batch.
        """
    # Separate images and targets from the batch
    _images = [item[0] for item in batch]
    bboxes = [item[1][0] for item in batch]
    obj_scores = [item[1][1] for item in batch]
    class_labels = [item[1][2] for item in batch]

    # Stack images into a single tensor
    _images = torch.stack(_images, dim=0)

    # Concatenate bounding boxes, object scores, and class labels
    bbox_targets = torch.cat(bboxes, dim=0)
    obj_targets = torch.cat(obj_scores, dim=0)
    class_targets = torch.cat(class_labels, dim=0)

    return _images, (bbox_targets, obj_targets, class_targets)


# DataLoader
train_dataset = YOLODataset('DataSet/train', 'DataSet/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,collate_fn=collate_fn)

test_dataset = YOLODataset('DataSet/test', 'DataSet/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,collate_fn=collate_fn)


# Initialize model, loss function, and optimizer
numClasses = 1
numAnchors = 3
model = YOLOv3(numClasses=numClasses, numAnchors=numAnchors)
criterion = YOLOLoss(numAnchors, numClasses, 13)
optimizer = Adam(model.parameters(), lr=1e-4)


# Training loop
numEpochs = 10
for epoch in range(numEpochs):
    model.train()
    runningLoss = 0.0
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()

    # print(f'Epoch {epoch+1}/{numEpochs}, Loss: {runningLoss/len(train_loader)}')

# # Test model
# model.eval()
# with torch.no_grad():
#     for images, targets in test_loader:
#         outputs = model(images)
#         # Post-processing would be done here, such as applying NMS
#         # Example: print output shapes
#         for out in outputs:
#             print(out.shape)