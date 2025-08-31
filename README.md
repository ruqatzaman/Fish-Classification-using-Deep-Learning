# Fish Image Classification (PyTorch CNN)

This repository contains a **PyTorch** implementation of a Convolutional Neural Network (CNN) that classifies fish images into species. The project is organized around a Jupyter notebook (`fish-classification-465.ipynb`) that trains a small CNN from scratch on a folder-structured dataset.

> **Highlights**
>
> - 31-class image classification (configurable)
> - Lightweight CNN trained from scratch (no pretrained backbone)
> - Standard data augmentations and ImageNet-style normalization
> - Training & validation curves, evaluation metrics, confusion matrix
> - Save & load model via `state_dict` for inference

---

##  Dataset & Folder Structure

The notebook expects an **image folder dataset** split into `train`, `val`, and `test`, where **each class is a subfolder** containing its images. For example:

```
data/
└── FishImgDataset/
    ├── train/
    │   ├── Gold Fish/
    │   ├── Salmon/
    │   └── ...
    ├── val/
    │   ├── Gold Fish/
    │   ├── Salmon/
    │   └── ...
    └── test/
        ├── Gold Fish/
        ├── Salmon/
        └── ...
```

Update the paths at the top of the notebook (replace with your local paths):

```python
TRAIN_PATH = 'data/FishImgDataset/train'
VAL_PATH   = 'data/FishImgDataset/val'
TEST_PATH  = 'data/FishImgDataset/test'
```

> The number of classes is inferred from the subfolders in `TRAIN_PATH`.

---

##  Model Overview

A compact CNN defined in `FishClassificationCNN`:

- **Input**: RGB images resized to **128×128**
- **Feature extractor** (3 blocks):
  - Conv(3→32, 3×3, padding=1) → ReLU → MaxPool(2)
  - Conv(32→64, 3×3, padding=1) → ReLU → MaxPool(2)
  - Conv(64→128, 3×3, padding=1) → ReLU → MaxPool(2)
- **Classifier**:
  - Flatten (**128 × 16 × 16**)
  - Linear → 512 → ReLU → **Dropout(0.5)**
  - Linear → `num_classes`
- **Loss/Opt**: CrossEntropyLoss + Adam(lr=1e-3)

> `num_classes = len(train_dataset.classes)` — automatically set from your data.

---

##  Setup

Tested with **Python 3.10+**. Install dependencies:

```bash
pip install torch torchvision torchaudio              numpy pandas matplotlib seaborn scikit-learn pillow
```

Optional (for Jupyter):
```bash
pip install jupyter notebook
```

> The notebook auto-detects CUDA (`cuda` if available, otherwise `cpu`).

---

##  Training (in Notebook)

Open the notebook and run cells in order:

1. **Configure paths & transforms** (resize to 128×128, random flip/rotation, normalization).
2. **Create datasets & loaders** (batch size **32**, shuffle train).
3. **Define model**: `FishClassificationCNN(num_classes=...)`
4. **Train** for `num_epochs=10` (editable):
   ```python
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=1e-3)

   train_losses, train_accuracies, val_losses, val_accuracies = train_model(
       model, train_loader, val_loader, criterion, optimizer, num_epochs=10
   )
   ```
5. **Plot curves**:
   ```python
   plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)
   ```

---

##  Evaluation

The notebook evaluates on the **test** split and prints:

- Overall **test accuracy**
- **Classification report** (per-class precision/recall/F1)
- **Macro F1 score**
- **Confusion matrix** heatmap

Example call:
```python
evaluate_model(model, test_loader, train_dataset.classes)
```

---

##  Save &  Load

**Save weights** (state_dict only):
```python
torch.save(model.state_dict(), "CNN_fish.pth")
```

**Load for inference** (two compatible options):

```python
# Option A (PyTorch 2.0+ sometimes supports weights_only)
state = torch.load("CNN_fish.pth", weights_only=True)

# Option B (safe & common):
state = torch.load("CNN_fish.pth", map_location="cpu")

model = FishClassificationCNN(num_classes=len(train_dataset.classes))
model.load_state_dict(state)
model.eval()
```

> If `weights_only=True` raises an error in your environment, use **Option B**.

---

##  Inference on a Single Image

Use the same normalization as training (no random augmentations):

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

transform_single_image = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image.to(device)

def predict_image(image_path, model, transform, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = preprocess_image(image_path, transform, device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]
```

Usage:
```python
cls = predict_image("path/to/image.jpg", model, transform_single_image, train_dataset.classes)
print("Predicted:", cls)
```

---

##  Project Structure (example)

```
.
├── fish-classification-465.ipynb     
├── CNN_fish.pth                     
└── data/
    └── FishImgDataset/
        ├── train/ ...               
        ├── val/   ...
        └── test/  ...
```

##  Acknowledgements

- Built with **PyTorch** & **Torchvision**.
- Dataset expected in an **ImageFolder-style** split (`train/val/test` with class subfolders).

