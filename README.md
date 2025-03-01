![image](https://github.com/user-attachments/assets/ae35c545-b334-4f69-8d7a-c1b95626c03a)# DST_PRED_ZERO

## Project Overview
DST_PRED_ZERO is a project focused on forecasting the Disturbance Storm Time (DST) Index using deep learning. The goal is to enhance space weather prediction by accurately forecasting geomagnetic storms.

## Introduction
The Dst-index (Disturbance Storm Time index) is a key measure of geomagnetic activity, representing the strength of Earth's ring current. It is calculated based on the average horizontal magnetic field observed at four low-latitude geomagnetic observatories. When the Dst-index decreases (more negative values), it indicates the presence of a geomagnetic storm, which can disrupt Earth's magnetosphere. These storms are caused by solar wind disturbances, such as coronal mass ejections (CMEs) and high-speed solar wind streams, which interact with Earth’s magnetic field.

Predicting the Dst-index is important because geomagnetic storms can have serious effects on technology and infrastructure. They can cause power grid failures, damage satellites, interfere with GPS signals, and affect radio communications. In extreme cases, strong geomagnetic storms can even increase radiation exposure for astronauts and high-altitude flights. By developing accurate models to predict the Dst-index, scientists can improve space weather forecasting and help prevent damage to critical systems. This project focuses on selecting the most relevant features from solar wind and interplanetary magnetic field (IMF) data to build a reliable predictive model for the Dst-index.

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

