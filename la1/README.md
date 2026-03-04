
# Multilayer Perceptron in C++

This project is an implementation of a **Multilayer Perceptron (MLP)** neural network written in C++.  
The network is trained using **online backpropagation with momentum** and can be configured from the command line.

The project was developed for educational purposes.

---


## Project Structure

```text
.
├── Makefile
├── README.md
├── la1.cpp
├── imc/
│   ├── MultilayerPerceptron.cpp
│   ├── MultilayerPerceptron.h
│   ├── util.cpp
│   └── util.h
└── dat/
    ├── train_xor.dat
    └── test_xor.dat
```


## Build

To compile the project, make sure you have `g++` and `make` installed.

`make` This will create the executable:  bin/la1

---


## Usage

The program is executed from the command line and accepts the following arguments (order does not matter):

-t <file> : training dataset (required)

-T <file> : test dataset (optional, training data is used if skipped)

-i <int> : number of training iterations (default: 1000)

-l <int> : number of hidden layers (default: 1)

-h <int> : number of neurons per hidden layer (default: 5)

-e <float>: learning rate eta (default: 0.1)

-m <float>: momentum mu (default: 0.9)

-s : normalize input and output data

Example (XOR problem)
./bin/la1 -t dat/train_xor.dat -T dat/test_xor.dat -i 1000 -h 32

---


## Output

During training, the program prints:

- training error for each iteration

- final network weights

- comparison between desired and obtained outputs on the test set

- final training and test mean squared error (MSE)



## Author

Kamil Staniszewski

=======
# Multilayer Perceptron in C++

This project is an implementation of a **Multilayer Perceptron (MLP)** neural network written in C++.  
The network is trained using **online backpropagation with momentum** and can be configured from the command line.

The project was developed as part of an academic assignment and is intended for educational purposes.

---


## Project Structure

.
├── Makefile
├── README.md
├── la1.cpp
├── imc/
│ ├── MultilayerPerceptron.cpp
│ ├── MultilayerPerceptron.h
│ ├── util.cpp
│ └── util.h
├── dat/
│ ├── train_xor.dat
│ └── test_xor.dat

---


## Build

To compile the project, make sure you have `g++` and `make` installed.

`make` This will create the executable:  bin/la1

---


## Usage

The program is executed from the command line and accepts the following arguments (order does not matter):

-t <file> : training dataset (required)

-T <file> : test dataset (optional, training data is used if skipped)

-i <int> : number of training iterations (default: 1000)

-l <int> : number of hidden layers (default: 1)

-h <int> : number of neurons per hidden layer (default: 5)

-e <float>: learning rate eta (default: 0.1)

-m <float>: momentum mu (default: 0.9)

-s : normalize input and output data

Example (XOR problem)
./bin/la1 -t dat/train_xor.dat -T dat/test_xor.dat -i 1000 -h 32

---


## Output

During training, the program prints:

- training error for each iteration

- final network weights

- comparison between desired and obtained outputs on the test set

- final training and test mean squared error (MSE)



Author
Kamil Staniszewski
>>>>>>> 90bcb9f (Porządkowanie plików i dodanie la2):la1/README.md
