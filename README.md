# Facetell

# Structure
```Facetell/
│
├── data/
│   ├── faces_224/
│   └── metadata.csv
│
├── models/
│   └── xception_model.keras
│
├── logs/
│   └── training_log.txt
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── model.py
│   └── evaluate.py
│
├── config.yaml
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```
# DeepFake Detection Model

This project uses two distinct datasets of faces consisting of deepfaked images and unaltered images which are given the labels REAL (if unaltered) and FAKE (if manipulated). We are using Kaggle's deepfake_faces dataset (https://www.kaggle.com/datasets/dagnelies/deepfake-faces) as well as finetuning the Xception model for deepfake classification. We trained the model using tensorflow, keras and other ML libraries and trained the model using the A100 GPU accelerator, which significantly sped up the training process. We also evaluated the model on benchmarks such as accuracy, precision, recall, F1-score, and AUC and it passed all the tests. The final trained model's file is 87MB and we cannot upload it here as it is too big for Git. But if you finish training the model, you can use the notebook script we wrote (Model_test.ipynb) to test out the model with your own custom data images. 

Here are some specs for the model:
- Model Architecture: We used the Xception architecture, which is a convolutional neural network designed for image classification. It leverages depthwise separable convolutions to reduce computational complexity while maintaining high performance.

- Input Size: The model expects input images of size 224x224 pixels with three color channels (RGB).

- Number of Parameters: The model consists of ~22.9 million parameters, making it highly capable of capturing complex patterns in image data.

- Output: The model outputs a single probability value, where:

Values close to 1.0 indicate that the image is likely FAKE.
Values close to 0.0 indicate that the image is likely REAL.
Loss Function: We used binary cross-entropy as the loss function, which is common for binary classification tasks.

- Optimizer: The model was trained using the Adam optimizer, with a learning rate of 1e-4, which was fine-tuned to achieve optimal performance.

- Batch Size: We trained the model with a batch size of 32, allowing for efficient utilization of GPU memory while maintaining fast convergence.

- Data Augmentation: We applied various data augmentation techniques such as random rotation, flipping, and zooming to increase the diversity of the training data and improve model generalization.

- Training Duration: The model was trained for 10 epochs on an A100 GPU, which reduced training time significantly compared to CPUs or lower-end GPUs.

- Evaluation Metrics: We evaluated the model using standard metrics for binary classification:

   - Accuracy: The percentage of correct predictions (REAL or FAKE) on the test set.
   - Precision: How many of the predicted FAKE images were actually FAKE.
   - Recall: How many of the actual FAKE images were correctly identified.
   - F1-Score: The harmonic mean of precision and recall.
   - Performance: On the evaluation dataset, the model achieved:
      - Accuracy: ~95%
      - Precision: ~94%
      - Recall: ~96%
      - F1-Score: ~95%

## Setup and Installation (for local setup)
Note: For Colab, see below this local setup section

1. Clone the repository:
   ```
   git clone https://github.com/sia1408/Facetell.git
   cd Facetell
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Setup for Colab
```git clone https://github.com/sia1408/Facetell.git```
```%cd Facetell/```
```!pip install -r requirements.txt```
```!python main.py --mode train --config config.yaml```
```%run src/evaluate.py #if you wanna evaluate the model's performance```

## Data Preparation

1. Place your dataset in the `data/` directory (if you have any special data, otherwise stick to data ive uploaded)
   - Put face images (.jpg) in `data/faces_224/`
   - Ensure `data/metadata.csv` is present with correct labels

2. Review and adjust the `config.yaml` file if necessary to match your data paths and desired parameters.

## Running the Program

### Training the Model

1. To train the model, run:
   ```
   python main.py --mode train
   ```
   This will train the model using the parameters specified in `config.yaml` and save the trained model in the `models/` directory.

### Evaluating the Model

2. After training, to evaluate the model on the test set, run:
   ```
   python main.py --mode evaluate
   ```
   This will load the trained model and evaluate its performance, printing out metrics such as accuracy, classification report, and confusion matrix.

## Additional Notes

- The trained model will be saved as `models/xception_model.keras`
- You can modify hyperparameters and file paths in `config.yaml`

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are correctly installed
2. Check that your data is in the correct format and location
3. Verify that the paths in `config.yaml` are correct for your system

For any further questions or issues, please open an issue on the GitHub repository.
