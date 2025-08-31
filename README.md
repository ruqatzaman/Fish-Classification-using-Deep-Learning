A compact **CNN-based image classifier** for fish species built with **PyTorch** and **torchvision**.  
The project trains on an image dataset arranged as `train/`, `val/`, and `test/`, evaluates with a **classification report**, **confusion matrix**, and **multi-class ROC–AUC**, and includes a simple **single-image prediction** helper. The trained weights are saved to `customCNN_fish.pth`.

> **Status:** Notebook-first workflow. Run all cells in order.

---
- Custom CNN with 3 conv blocks + 2 fully connected layers
- Data augmentation (resize, random flips, small affine jitter)
- Training curves (loss & accuracy)
- Evaluation on a held-out test set (classification report, confusion matrix, ROC–AUC (OvO))
- Single-image inference helper
- Model checkpoint saving (`.pth`)

---
The notebook expects an ImageFolder-style dataset (each class is a subfolder containing images):

```
FishImgDataset/
├── train/
│   ├── CLASS_1/
│   ├── CLASS_2/
│   └── ...
├── val/
│   ├── CLASS_1/
│   ├── CLASS_2/
│   └── ...
└── test/
    ├── CLASS_1/
    ├── CLASS_2/
    └── ...
```

By default, the notebook paths are set for Kaggle (adjust for local use):
```python
train_dir = "/kaggle/input/fish-dataset/FishImgDataset/train"
val_dir   = "/kaggle/input/fish-dataset/FishImgDataset/val"
test_dir  = "/kaggle/input/fish-dataset/FishImgDataset/test"
```

---

**Input:** 3×128×128 images  
**Backbone:** 3 convolutional blocks with ReLU and 2×2 MaxPool  
**Head:** Flatten → Linear(32768→512) → ReLU → Dropout(0.5) → Linear(512→`num_classes`)

```text
[Conv2d(3→32, k3, p1)] → ReLU → MaxPool(2)
 → [Conv2d(32→64, k3, p1)] → ReLU → MaxPool(2)
 → [Conv2d(64→128, k3, p1)] → ReLU → MaxPool(2)
 → Flatten(128×16×16 = 32768) → Linear(32768→512) → ReLU → Dropout(0.5) → Linear(512→num_classes)
```

**Loss:** CrossEntropyLoss  
**Optimizer:** Adam(lr=1e-3)  
**Batch size:** 32  
**Epochs (default in notebook):** 10  

---

Install the core dependencies (CPU-only shown below; visit pytorch.org for CUDA builds):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib pillow seaborn
```

> If you're on Kaggle, most packages are preinstalled.

---

Training-time transforms (resize to 128×128, random horizontal flip, small affine jitter):
```python
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, shear=0.2, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
])
```

Validation/Test transforms:
```python
transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # see note below
])
```


---

## How to run (Kaggle or local)

1. **Clone/download** this repository and place your dataset in the structure shown above.  
2. Open `fish-classification465.ipynb` in Jupyter/VS Code/Kaggle.  
3. Update the dataset paths if needed:
   ```python
   train_dir = "path/to/FishImgDataset/train"
   val_dir   = "path/to/FishImgDataset/val"
   test_dir  = "path/to/FishImgDataset/test"
   ```
4. (Optional) Adjust hyperparameters in the notebook:
   ```python
   batch_size = 32
   epochs = 10
   lr = 0.001
   ```
5. **Run all cells in order.** The notebook will:
   - Train the CNN and print per-epoch loss/accuracy
   - Plot **training/validation curves**
   - Evaluate on the test set and print a **classification report**
   - Show a **confusion matrix** heatmap
   - Compute **overall ROC–AUC (OvO)** across classes
   - Save the model weights as `customCNN_fish.pth`

---

The notebook includes a helper to predict a single image’s class:

```python
from PIL import Image

transform_single_image = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # keep in sync with training
])

predicted_class = predict_image(
    image_path="path/to/your_image.jpg",
    model=model,
    transform=transform_single_image,
    class_names=train_data.classes
)

print("Predicted Class:", predicted_class)
```

> Ensure the `model` is in `eval()` mode and weights are loaded if predicting in a fresh session.

---

## Outputs you’ll see

- `classification_report` (precision, recall, f1-score per class + accuracy)
- `confusion_matrix` heatmap (via seaborn)
- Overall **ROC–AUC (one-vs-one)** across classes
- Saved weights: `customCNN_fish.pth`

> Exact scores depend on your dataset split, augmentation, and training duration.

---

- **Normalize channels correctly:** Use 3 values for RGB as shown above.
- **GPU:** Training is faster with CUDA. The notebook auto-selects `cuda` if available.
- **Class imbalance:** Consider `WeightedRandomSampler` or class-weighted loss if needed.
- **Overfitting:** Increase augmentation, add dropout, or reduce model capacity.
- **Underfitting:** Train for more epochs, raise model capacity, or lower regularization.
- **Reproducibility:** Set seeds (`torch`, `numpy`, `random`) if you need stable runs.

---

## Acknowledgements

- Built with **PyTorch**, **torchvision**, **scikit-learn**, **matplotlib**, **Pillow**, and **seaborn**.
- Dataset prepared in ImageFolder format with class subdirectories.
