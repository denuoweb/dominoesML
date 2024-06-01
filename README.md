
# Domino Dot Counter

## Introduction

This repository contains three distinct machine learning implementations aimed at counting the number of dots on dominoes. Each implementation uses a different machine learning framework: TensorFlow/Keras, Scikit-learn, and PyTorch. The primary challenge with these models is the limited number of training images, which constrains their accuracy. This README provides comprehensive instructions on setting up, running, and understanding each model.

## Repository Structure

The repository is structured as follows:

```
basePhotos/
    ├── generated/
    ├── test/
    ├── train/
    └── validate/
logs/
    ├── train/
    └── validation/
appKeras.py
appSci.py
appTorch.py
best_model.h5
best_model.keras
domino_dot_counter.keras
requirements.txt
```

- **basePhotos/**: Contains subdirectories for training, validation, and test images.
- **logs/**: Contains subdirectories for storing training and validation logs.
- **appKeras.py**: TensorFlow/Keras implementation.
- **appSci.py**: Scikit-learn implementation.
- **appTorch.py**: PyTorch implementation.
- **best_model.h5**: Model saved from TensorFlow/Keras.
- **best_model.keras**: Another Keras model.
- **domino_dot_counter.keras**: Another Keras model.
- **requirements.txt**: List of dependencies required to run the project.

## Setup and Installation

### Prerequisites

- Python 3.6 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Ensure that your images are organized in the following directory structure under `basePhotos`:

- `basePhotos/train`: Contains training images.
- `basePhotos/validate`: Contains validation images.
- `basePhotos/test`: Contains test images.

The filenames should start with the number of dots they represent, followed by an underscore or directly a dot (e.g., `3_1.jpg` or `5.jpg`).

## Model Descriptions

### 1. TensorFlow/Keras Model

The TensorFlow/Keras model is implemented in `appKeras.py`. This model utilizes a Convolutional Neural Network (CNN) with data augmentation layers.

#### Detailed Workflow

1. **Data Parsing and Preprocessing**:
   - Images are read, decoded, resized, and normalized.
   - Labels are extracted from filenames.

2. **Dataset Creation**:
   - Training, validation, and test datasets are created using `tf.data.Dataset`.

3. **Data Augmentation**:
   - Random rotations, zooms, and flips are applied to augment the data during training.

4. **Model Architecture**:
   - The model consists of multiple convolutional layers followed by max pooling layers.
   - A dropout layer is included to prevent overfitting.
   - The final layer is a dense layer with a single output for regression.

5. **Training**:
   - The model is trained for 5000 epochs.
   - Checkpoints, TensorBoard logging, and early stopping are used as callbacks.

6. **Evaluation**:
   - The model is evaluated on the test dataset.

#### Training the Model

To train the TensorFlow/Keras model, run:
```bash
python appKeras.py
```

### 2. Scikit-learn Model

The Scikit-learn model is implemented in `appSci.py`. This model employs a Decision Tree Classifier.

#### Detailed Workflow

1. **Data Loading**:
   - Images are loaded and flattened.
   - Labels are extracted from filenames.

2. **Model Training**:
   - StratifiedKFold is used for cross-validation.
   - The Decision Tree Classifier is trained on the combined training and validation datasets.

3. **Evaluation**:
   - Cross-validation scores are calculated and printed.

#### Training the Model

To train the Scikit-learn model, run:
```bash
python appSci.py
```

### 3. PyTorch Model

The PyTorch model is implemented in `appTorch.py`. This model uses a pre-trained ResNet18, modified for regression to predict the number of dots.

#### Detailed Workflow

1. **Dataset Class**:
   - A custom dataset class `DominoDotsDataset` is defined.
   - Labels are extracted from filenames.

2. **Data Transformations**:
   - Images are resized, randomly flipped and rotated, and normalized.

3. **Model Architecture**:
   - The last layer of ResNet18 is modified to output a single value for regression.

4. **Training**:
   - The model is trained for 100 epochs.
   - A custom loss function is defined to penalize prediction errors.

5. **Evaluation**:
   - The model is evaluated on the test dataset.

#### Training the Model

To train the PyTorch model, run:
```bash
python appTorch.py
```

## Running TensorBoard

For the TensorFlow/Keras model, you can visualize the training process using TensorBoard. Run:
```bash
tensorboard --logdir=./logs
```

## Future Considerations

- **Data Augmentation**: Further augmentation techniques can be explored to generate more diverse training samples.
- **Advanced Models**: Explore more advanced architectures like EfficientNet or transfer learning with fine-tuning.
- **Hyperparameter Tuning**: Perform hyperparameter tuning to optimize model performance.
- **Ensemble Methods**: Combine predictions from multiple models to improve accuracy.
- **Deployment**: Implement model deployment pipelines for real-time predictions.

## Conclusion

This project demonstrates the application of different machine learning frameworks to solve the problem of counting dots on dominoes. Despite the challenge of limited training data, these implementations provide a foundation for further development and improvement.

## Contact

For any queries or contributions, please reach out to the repository maintainers.

---

**Note**: Ensure that you have the appropriate permissions and follow data privacy guidelines while handling the images and model files.
