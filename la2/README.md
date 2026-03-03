# Multilayer Perceptron in C++ – Classification Version (la2)

This project is an extended implementation of a **Multilayer Perceptron (MLP)** neural network written in C++.

Unlike the previous version, this implementation is designed for **multi-class classification problems** and includes:

- Online and offline training
- Softmax output layer
- Cross-Entropy and MSE loss functions
- Classification accuracy (CCR)
- Confusion matrix analysis
- Kaggle-style prediction mode

The project was developed for educational purposes.

---

## Project Structure


.
├── Makefile
├── README.md
├── la2.cpp
├── imc/
│ ├── MultilayerPerceptron.cpp
│ ├── MultilayerPerceptron.h
│ ├── util.cpp
│ └── util.h
├── dat/
│ ├── train_xor.dat
│ ├── test_xor.dat
│ ├── train_nomnist.dat
│ └── test_nomnist.dat
│ ├── train_quake.dat
│ └── test_quake.dat


---

## Build

To compile the project, make sure you have `g++` and `make` installed.

`make` This will create the executable:  bin/la1

---

## Usage

The program supports two modes:

---

### Training / Evaluation Mode (default)


./bin/la2 -t <train.dat> [options]


Arguments:

-t <file> : training dataset (required)

-T <file> : test dataset (optional, training data is used if skipped)

-i <int> : number of training iterations (default: 1000)

-l <int> : number of hidden layers (default: 1)

-h <int> : neurons per hidden layer (default: 5)

-e <float> : learning rate (default: 0.1)

-m <float> : momentum (default: 0.9)

-o : enable online training

-f <0|1> : error function (0 = MSE, 1 = Cross-Entropy)

-s : use Softmax in output layer

-n : normalize input data to [-1,1]

-w <file> : save best weights

Example (XOR classification)
./bin/la2 -t dat/train_xor.dat -T dat/test_xor.dat -i 500 -h 8 -o -s -f 1 -n

---

### Prediction Mode (Kaggle-style)

After training and saving weights:

./bin/la2 -p -w best_weights.txt -T dat/test_nomnist.dat

This mode:
- Loads previously saved weights
- Performs classification
- Generates predictions for submission

---

## Output

During execution, the program prints:

- Training and test error
- Classification accuracy (CCR)
- Confusion matrix
- Misclassified examples
- Final mean and standard deviation over 5 different random seeds

Example metrics:

- Train/Test Error (Mean ± SD)
- Train/Test CCR (Mean ± SD)

---


Author
Kamil Staniszewski