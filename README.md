# Egg Fertility Prediction Model 

A deep learning project using VGG16 transfer learning to classify egg fertility images as **Fertile (FER)** or **Infertile (INF)**.

## Project Overview

This project implements a Convolutional Neural Network (CNN) based on the pre-trained VGG16 architecture to predict egg fertility from images. The model uses transfer learning with ImageNet weights and fine-tuning for binary classification.

## Dataset

- **Source**: [Kaggle - Egg Fertility Dataset](https://www.kaggle.com/datasets/mostefatabbakh/egg-fertility-1275)
- **Classes**: 
  - Fertile (FER)
  - Infertile (INF)
- **Structure**:
  ```
  Dataset/
  ├── train/
  │   ├── FER/
  │   └── INF/
  ├── valid/
  │   ├── FER/
  │   └── INF/
  └── test/
      ├── FER/
      └── INF/
  ```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Kagglehub
- NumPy, Matplotlib

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PriscillajospinG/Egg-fertility-prediction-model.git
   cd Egg-fertility-prediction-model
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow kagglehub numpy matplotlib
   ```

3. **Download the dataset** (automatically handled in the notebook):
   ```python
   import kagglehub
   path = kagglehub.dataset_download("mostefatabbakh/egg-fertility-1275")
   ```

##  Model Architecture

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense (256 units, ReLU activation)
  - Dropout (0.5)
  - Dense (1 unit, Sigmoid activation)
- **Input Size**: 224x224x3
- **Output**: Binary classification (FER/INF)

##  Data Augmentation

The training data is augmented with:
- Rotation (±30°)
- Width/Height shift (20%)
- Shear transformation (20%)
- Zoom (±30%)
- Brightness adjustment (0.7-1.3)
- Horizontal and vertical flips

##  Training

- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Binary Cross-Entropy
- **Callbacks**:
  - EarlyStopping (patience=5)
  - ModelCheckpoint (saves best model)
- **Epochs**: Up to 30 (with early stopping)
- **Batch Size**: 32

##  Results

The model's performance is evaluated on the test set, with accuracy metrics displayed in the notebook. Training history plots show accuracy and loss curves for both training and validation sets.

##  Model Files

- `best_model.h5`: Saved model with best validation performance
- `VGG16.ipynb`: Complete training and prediction notebook

##  Usage

### Training the Model

Run all cells in `VGG16.ipynb` sequentially to:
1. Download and prepare the dataset
2. Build and train the VGG16 model
3. Evaluate on test data
4. Save the best model

### Making Predictions

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("best_model.h5")

# Load and preprocess an image
img_path = "path/to/your/image.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
label = "Fertile (FER)" if pred[0][0] > 0.5 else "Infertile (INF)"
print(f"Prediction: {label}")
```

## 📁 Project Structure

```
Egg-fertility-prediction-model/
├── VGG16.ipynb          # Main notebook with training and prediction code
├── best_model.h5        # Trained model weights
├── README.md            # Project documentation
└── Dataset/             # Dataset folder (train/valid/test splits)
    ├── train/
    ├── valid/
    └── test/
```


## 👤 Author

**Priscilla Jospin G**
- GitHub: [@PriscillajospinG](https://github.com/PriscillajospinG)

## 🙏 Acknowledgments

- Dataset provided by [Mostefa Tabbakh on Kaggle](https://www.kaggle.com/datasets/mostefatabbakh/egg-fertility-1275)
- VGG16 architecture from the ImageNet challenge
- TensorFlow and Keras teams for the excellent deep learning framework

---

**Note**: This model is for educational and research purposes. For production use in agriculture or poultry farming, additional validation and testing are recommended.

Made with ❤️ Priscilla
