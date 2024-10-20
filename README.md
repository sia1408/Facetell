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

This project implements a deep learning model for detecting deepfake images using a Xception-based architecture.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
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

## Data Preparation

1. Place your dataset in the `data/` directory:
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