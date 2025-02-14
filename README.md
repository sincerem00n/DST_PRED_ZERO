# DST_PRED_ZERO

## Project Overview
DST_PRED_ZERO is a project focused on forecasting the Disturbance Storm Time (DST) Index using advanced data science techniques. The goal is to enhance space weather prediction by accurately forecasting geomagnetic storms.

## Features
- **Data Collection**: Aggregates data from various space weather sources.
- **Data Processing**: Cleans and preprocesses the data for analysis.
- **Model Training**: Utilizes machine learning algorithms to forecast the DST Index.
- **Evaluation**: Assesses the performance of the forecasting models.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To train the forecasting model, use the following command:
```bash
python train.py
```
### Command Line Arguments

The training script accepts several command line arguments:

- `--dir`: Directory for saving the trained model (default: `results`)
- `--lr`: Learning rate (default: `0.0005`)
- `--epochs`: Number of epochs (default: `100`)
- `--batch_size`: Batch size (default: `32`)

Example usage:
```bash
python train.py --dir results --lr 0.0001 --epochs 50 --batch_size 64
```

## Contributing
We welcome contributions! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For any questions or feedback, please contact [jingjaijan62@gmail.com].

