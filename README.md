
# Shakespeare vs. Non-Shakespeare Texts Prediction

## Overview

This project focuses on building a machine learning model to classify whether a given poem is written by William Shakespeare or not. It utilizes Aspect-Based Sentiment Analysis (ABSA) techniques and employs BERT (Bidirectional Encoder Representations from Transformers) as the underlying model architecture. The model is trained on datasets containing normalized poems attributed to either Shakespeare or non-Shakespeare authors.

## Project Structure

The project is organized into several components:

- **src**: Contains the source code for the ABSA model training, validation, and prediction.
- **dataset**: Stores the datasets used for training and testing the model.
- **pretrained_models**: Contains the pre-trained BERT models used for aspect-based sentiment analysis.
- **results**: Stores the output files generated during model training, validation, and prediction.

## Requirements

- Python 3.x
- PyTorch
- Transformers library (Hugging Face)
- NLTK (Natural Language Toolkit)

## Setup

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/shakespeare-texts-prediction.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

To train the ABSA model on a new dataset:

1. Prepare your dataset by storing poems in CSV format under the `dataset` directory.
2. Update the dataset paths in the training script if necessary.
3. Run the training script:

   ```bash
   python train_model.py
   ```

4. The trained model will be saved in the `pretrained_models` directory.

## Validation

To validate the trained model:

1. Prepare a test dataset containing poems for validation.
2. Update the dataset path in the validation script if necessary.
3. Run the validation script:

   ```bash
   python validate_model.py
   ```

4. Performance metrics such as accuracy and loss will be displayed.

## Prediction

To make predictions on a new set of poems:

1. Load the trained model using the appropriate script.
2. Prepare the poems to be predicted and store them in CSV format under the `dataset` directory.
3. Update the dataset path in the prediction script if necessary.
4. Run the prediction script:

   ```bash
   python predict_texts.py
   ```

5. Predictions will be generated and saved in the `results` directory.

## Contributing

Contributions are welcome! If you'd like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure all tests pass.
4. Submit a pull request detailing your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README with additional details, usage instructions, or any other relevant information about your project!
