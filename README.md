# Object Recognition Project

This project is a work in progress for object recognition using PyTorch and the COCO dataset. The main script, `objectRecognition.py`, demonstrates how to train and evaluate a Faster R-CNN model for detecting objects in images.

<img src="https://github.com/user-attachments/assets/c3f4390f-8748-4e87-8218-950f919da68f" alt="Sample" width="400" />

<img src="https://github.com/user-attachments/assets/70803e6b-26e3-4729-83ca-cbbe49a6a01d" alt="Sample" width="400" />

<img src="https://github.com/user-attachments/assets/b41415bc-2d45-4884-8a99-6aae5fdc8b08" alt="Sample" width="400" />


## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- PIL (Pillow)

## Dataset

The project uses the COCO dataset for training and validation. Update the paths to the COCO dataset in the script:

```python
# Define paths for COCO dataset
coco_train_images = "/data/coco/train2017/"
coco_train_annotations = "/data/coco/annotations/instances_train2017.json"
coco_val_images = "/data/coco/val2017/"
coco_val_annotations = "/data/coco/annotations/instances_val2017.json"
```

### Training

To train the model, run the script:

```sh
python objectRecognition.py
```

The script will:
1. Load the COCO dataset.
2. Define the transformations for the dataset.
3. Create custom dataset classes and data loaders.
4. Load a pre-trained Faster R-CNN model and modify it for custom classes.
5. Train the model for a specified number of epochs.
6. Save checkpoints after each epoch.

### Evaluation

The script also includes code for evaluating the model on the validation set and visualizing predictions. After training, the model is set to evaluation mode, and predictions are visualized on a sample image from the validation set.

## Visualization

The script includes functions to visualize predictions with bounding boxes and labels on images using `matplotlib`.

## Future Work
- Improve the quality of the existing code.
- Improve the training loop with more epochs and better hyperparameters.
- Add support for more custom classes.
- Implement more advanced data augmentation techniques.


