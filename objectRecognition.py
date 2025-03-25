import os
from matplotlib import pyplot as plt
import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import draw_bounding_boxes
from torchvision.datasets import CocoDetection
from torch.amp import GradScaler, autocast

# Print PyTorch and CUDA information
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Define paths for COCO dataset
coco_train_images = "/data/coco/train2017/"
coco_train_annotations = "/data/coco/annotations/instances_train2017.json"
coco_val_images = "/data/coco/val2017/"
coco_val_annotations = "/data/coco/annotations/instances_val2017.json"

# Define transformations for the dataset
transforms = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class for COCO
class CustomCocoDataset(CocoDetection):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # Extract bounding boxes and labels from COCO annotations
        boxes = []
        labels = []

        for obj in target:
            x_min, y_min, width, height = obj['bbox']
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj['category_id'])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return img, target

# Load train and validation datasets
train_dataset = CustomCocoDataset(
    root=coco_train_images,
    annFile=coco_train_annotations,
    transform=Compose([ToTensor()])
)

val_dataset = CustomCocoDataset(
    root=coco_val_images,
    annFile=coco_val_annotations,
    transform=Compose([ToTensor()])
)

# Collate function for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

# Load and preprocess a custom image
def load_custom_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device), image

# Visualize predictions on an image
def visualize_predictions(image, boxes, labels, scores, threshold=0.5):
    filtered_indices = scores > threshold
    boxes = boxes[filtered_indices]
    labels = labels[filtered_indices]
    scores = scores[filtered_indices]

    plt.imshow(image)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        )
        plt.text(x1, y1-10, f"Label: {label} Score: {score:.2f}", color='red',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # DataLoader for training and validation datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn 
    )

    # Print targets from the first batch of the training loader
    for images, targets in train_loader:
        print(targets)  
        break

    # Visualize bounding boxes on the first image of the first batch
    for images, targets in train_loader:
        if len(targets) == 0 or "boxes" not in targets[0] or "labels" not in targets[0]:
            print("Skipping batch due to missing or invalid annotations.")
            continue

        img = images[0].mul(255).byte() 
        boxes = targets[0]["boxes"]
        labels = targets[0]["labels"]

        labels = [str(label.item()) for label in labels]

        img_with_boxes = draw_bounding_boxes(img, boxes, labels=labels)
        to_pil_image(img_with_boxes).show()  
        break

    # Load pre-trained Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Modify the model for custom number of classes (background, person, car)
    num_classes = 3  # Background, person, car
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    print("Model device:", next(model.parameters()).device)

    # Define optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 1

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")

        # Print the current learning rate
        print("Learning Rate:", optimizer.param_groups[0]['lr'])

        model.train()  # Set the model to training mode
        epoch_loss = 0

        for images, targets in train_loader:
            # Move images and targets to the GPU
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()  # Zero out gradients

            with autocast(device_type="cuda"):  # Enable mixed precision
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()  # Backpropagation
            scaler.step(optimizer)  # Optimizer step
            scaler.update()

            epoch_loss += losses.item()

        # Step the learning rate scheduler
        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} completed. Loss: {epoch_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, f"checkpoint_epoch_{epoch+1}.pth")

    model.eval()  # Set model to evaluation mode

    # Evaluate on validation set and visualize predictions
    for images, targets in val_loader:
        print("Image device (validation):", images[0].device)  
        print("Target device (validation):", targets[0]['boxes'].device) 
        print("Model device (validation):", next(model.parameters()).device)
        images = [image.to(device) for image in images]
        outputs = model(images)

        # Visualize one example
        image = images[0].cpu()
        boxes = outputs[0]['boxes'].cpu().detach().numpy()
        labels = outputs[0]['labels'].cpu().detach().numpy()
        scores = outputs[0]['scores'].cpu().detach().numpy()

        visualize_predictions(to_pil_image(image), boxes, labels, scores)
        break  # Show one example

    # Custom Image Inference
    image_path = "assets/image_to_analyze.jpg"
    image_tensor, original_image = load_custom_image(image_path, device)

    model.eval()  # Ensure the model is in evaluation mode

    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract predictions
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    visualize_predictions(original_image, boxes, labels, scores)