
# Shakespeare vs. Non-Shakespeare Texts Prediction

## Overview

This project focuses on building a machine learning model to classify whether a given poem is written by William Shakespeare or not. It utilizes Aspect-Based Sentiment Analysis (ABSA) techniques and employs BERT (Bidirectional Encoder Representations from Transformers) as the underlying model architecture. The model is trained on datasets containing normalized poems attributed to either Shakespeare or non-Shakespeare authors.

## Project Structure

The project is organized into several components:

- **Final_Main**: Contains the source code for the ABSA model training, validation, and prediction.
- **csv_and_plots**: contains the functions that creates the datasets used for training, testing and predicting on the model.
- **absa.py**: Contains the classes that are used to train, validate and predict on the model.
- **requirments**: Contains the required libraries to import.
- **Train_dataset**: Contains the trained datasets to train aditional layer on the pre-trained model.
- **Related Files**: Contains related output and input files for example.
- **Prediction_dataset**: Contains csv Shakespeare poems to predict on the model.

## Requirements

- Python 3.x
- PyTorch
- Transformers library (Hugging Face)
- NLTK (Natural Language Toolkit)

## Setup

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/Final-Project.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
## Preparing the Datasets

to prepare the datasets to use them as input for the model:

1. prepare the poems of Shakespeare and non-Shakespeare at txt format.
2. open the csv_and_plots.ipynb
3. run the desired function, see that you are changing the paths and variables to fit your PC.
4. Validate that the output is at the same format as the similar file at the package

## Training the Model

To train additional layer on a pre-trained ABSA model on a new dataset:

1. Prepare your dataset in CSV format.
2. open Final_Main.py from the package locate the function that is used to train additional layer.
3. Run the training function.
4. The trained model will be saved and will be usefull for future prediction validation and work.

### Training Example:

Accuracy plot:

![Alt text](https://github.com/alonmodin1/Final-Project/blob/main/Part%20B/accuracy.png)

Loss plot:

![Alt text](https://github.com/alonmodin1/Final-Project/blob/main/Part%20B/loss.png)

## Validation

To validate the trained model:

1. Prepare a test dataset containing poems for validation.
2. open Final_Main.py from the package locate the function that is used to validate trained model.
3. Update the dataset and model path in the validation script if necessary.
4. Run the validation function.
5. Performance metrics such as accuracy and loss will be displayed.

### Validation Example:
![Alt text](https://github.com/alonmodin1/Final-Project/blob/main/Part%20B/vali.png)

## Prediction

To make predictions on a new set of poems:

1. open Final_Main.py from the package locate the function that is used to predic on a trained model.
2. Load the trained model using the appropriate path.
3. Prepare the poems to be predicted and store them in CSV format.
4. Update the dataset path in the prediction script.
5. Run the prediction function.
6. Predictions will be generated.

### Prediction Example:
![Alt text](https://github.com/alonmodin1/Final-Project/blob/main/Part%20B/pred.png)

## Contributers

* **Alon Modin**
* **Omer Asus**

Special thanks to Prof. Zeev Volkovich and Dr. Renata Avros for their guidance and support.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README with additional details, usage instructions, or any other relevant information about your project!
