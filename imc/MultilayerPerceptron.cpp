#include "MultilayerPerceptron.h"
#include "util.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <cmath>

using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: default parameter values
MultilayerPerceptron::MultilayerPerceptron()
{
    nOfLayers = 0;
    layers = NULL;

    // Default hyperparameters
    eta = 0.1;  // learning rate
    mu  = 0.9;  // momentum
}

// ------------------------------
// Allocate memory for the network data structures
// nl = number of layers, npl[] = number of neurons per layer
int MultilayerPerceptron::initialize(int nl, int npl[])
{
    nOfLayers = nl;
    layers = new Layer[nOfLayers];

    for (int l = 0; l < nOfLayers; ++l)
    {
        layers[l].nOfNeurons = npl[l];
        layers[l].neurons = new Neuron[layers[l].nOfNeurons];

        // Only hidden and output layers have weights
        if (l > 0)
        {
            for (int j = 0; j < layers[l].nOfNeurons; ++j)
            {
                // +1 for bias weight w[0]
                int nInputs = layers[l - 1].nOfNeurons + 1;

                layers[l].neurons[j].w          = new double[nInputs];
                layers[l].neurons[j].deltaW     = new double[nInputs];
                layers[l].neurons[j].lastDeltaW = new double[nInputs];
                layers[l].neurons[j].wCopy      = new double[nInputs];

                // Initialize accumulators to zero
                for (int k = 0; k < nInputs; ++k)
                {
                    layers[l].neurons[j].deltaW[k] = 0.0;
                    layers[l].neurons[j].lastDeltaW[k] = 0.0;
                }
            }
        }
    }

    return 1;
}

// ------------------------------
// Destructor
MultilayerPerceptron::~MultilayerPerceptron()
{
    freeMemory();
}

// ------------------------------
// Free memory of all allocated structures
void MultilayerPerceptron::freeMemory()
{
    if (layers == NULL) return;

    for (int l = 0; l < nOfLayers; ++l)
    {
        if (l > 0)
        {
            for (int j = 0; j < layers[l].nOfNeurons; ++j)
            {
                delete[] layers[l].neurons[j].w;
                delete[] layers[l].neurons[j].deltaW;
                delete[] layers[l].neurons[j].lastDeltaW;
                delete[] layers[l].neurons[j].wCopy;
            }
        }

        delete[] layers[l].neurons;
    }

    delete[] layers;
    layers = NULL;
    nOfLayers = 0;
}

// ------------------------------
// Initialize all weights randomly in [-1, 1]
void MultilayerPerceptron::randomWeights()
{
    for (int l = 1; l < nOfLayers; ++l)
    {
        for (int j = 0; j < layers[l].nOfNeurons; ++j)
        {
            int nInputs = layers[l - 1].nOfNeurons + 1; // +1 bias
            for (int k = 0; k < nInputs; ++k)
            {
                layers[l].neurons[j].w[k] = randomDouble(-1.0, 1.0);
            }
        }
    }
}

// ------------------------------
// Copy input vector into input layer outputs
void MultilayerPerceptron::feedInputs(double* input)
{
    for (int i = 0; i < layers[0].nOfNeurons; ++i)
    {
        layers[0].neurons[i].out = input[i];
    }
}

// ------------------------------
// Copy output layer values into output vector
void MultilayerPerceptron::getOutputs(double* output)
{
    int outLayer = nOfLayers - 1;
    for (int i = 0; i < layers[outLayer].nOfNeurons; ++i)
    {
        output[i] = layers[outLayer].neurons[i].out;
    }
}

// ------------------------------
// Save current weights into wCopy
void MultilayerPerceptron::copyWeights()
{
    for (int l = 1; l < nOfLayers; ++l)
    {
        for (int j = 0; j < layers[l].nOfNeurons; ++j)
        {
            int nInputs = layers[l - 1].nOfNeurons + 1;
            for (int k = 0; k < nInputs; ++k)
            {
                layers[l].neurons[j].wCopy[k] = layers[l].neurons[j].w[k];
            }
        }
    }
}

// ------------------------------
// Restore weights from wCopy
void MultilayerPerceptron::restoreWeights()
{
    for (int l = 1; l < nOfLayers; ++l)
    {
        for (int j = 0; j < layers[l].nOfNeurons; ++j)
        {
            int nInputs = layers[l - 1].nOfNeurons + 1;
            for (int k = 0; k < nInputs; ++k)
            {
                layers[l].neurons[j].w[k] = layers[l].neurons[j].wCopy[k];
            }
        }
    }
}

// ------------------------------
// Forward propagation using sigmoid activation
void MultilayerPerceptron::forwardPropagate()
{
    for (int l = 1; l < nOfLayers; ++l)
    {
        for (int j = 0; j < layers[l].nOfNeurons; ++j)
        {
            double net = layers[l].neurons[j].w[0]; // bias

            for (int i = 0; i < layers[l - 1].nOfNeurons; ++i)
            {
                net += layers[l].neurons[j].w[i + 1] * layers[l - 1].neurons[i].out;
            }

            layers[l].neurons[j].out = 1.0 / (1.0 + std::exp(-net));
        }
    }
}

// ------------------------------
// Compute Mean Squared Error for one pattern
double MultilayerPerceptron::obtainError(double* target)
{
    int outLayer = nOfLayers - 1;
    int nOutputs = layers[outLayer].nOfNeurons;

    double err = 0.0;
    for (int j = 0; j < nOutputs; ++j)
    {
        double diff = target[j] - layers[outLayer].neurons[j].out;
        err += diff * diff;
    }
    return err / nOutputs;
}

// ------------------------------
// Backpropagation of error (sigmoid derivative)
void MultilayerPerceptron::backpropagateError(double* target)
{
    int outLayer = nOfLayers - 1;

    // Output layer deltas
    for (int j = 0; j < layers[outLayer].nOfNeurons; ++j)
    {
        double out = layers[outLayer].neurons[j].out;
        layers[outLayer].neurons[j].delta = -(target[j] - out) * out * (1.0 - out);
    }

    // Hidden layers deltas (from last hidden to first hidden)
    for (int l = nOfLayers - 2; l >= 1; --l)
    {
        for (int j = 0; j < layers[l].nOfNeurons; ++j)
        {
            double sum = 0.0;

            for (int k = 0; k < layers[l + 1].nOfNeurons; ++k)
            {
                sum += layers[l + 1].neurons[k].w[j + 1] * layers[l + 1].neurons[k].delta;
            }

            double out = layers[l].neurons[j].out;
            layers[l].neurons[j].delta = sum * out * (1.0 - out);
        }
    }
}

// ------------------------------
// Accumulate gradient contributions into deltaW (one pattern)
void MultilayerPerceptron::accumulateChange()
{
    for (int l = 1; l < nOfLayers; ++l)
    {
        for (int j = 0; j < layers[l].nOfNeurons; ++j)
        {
            // Bias
            layers[l].neurons[j].deltaW[0] += layers[l].neurons[j].delta;

            // Weights from previous layer outputs
            for (int i = 0; i < layers[l - 1].nOfNeurons; ++i)
            {
                layers[l].neurons[j].deltaW[i + 1] += layers[l].neurons[j].delta * layers[l - 1].neurons[i].out;
            }
        }
    }
}

// ------------------------------
// Update weights using learning rate and momentum, then reset accumulators
void MultilayerPerceptron::weightAdjustment()
{
    for (int l = 1; l < nOfLayers; ++l)
    {
        for (int j = 0; j < layers[l].nOfNeurons; ++j)
        {
            int nInputs = layers[l - 1].nOfNeurons + 1;

            for (int i = 0; i < nInputs; ++i)
            {
                double change = eta * layers[l].neurons[j].deltaW[i]
                              + mu * eta * layers[l].neurons[j].lastDeltaW[i];

                layers[l].neurons[j].w[i] -= change;

                layers[l].neurons[j].lastDeltaW[i] = layers[l].neurons[j].deltaW[i];
                layers[l].neurons[j].deltaW[i] = 0.0;
            }
        }
    }
}

// ------------------------------
// One online SGD step for a single pattern
void MultilayerPerceptron::performEpochOnline(double* input, double* target)
{
    feedInputs(input);
    forwardPropagate();
    backpropagateError(target);
    accumulateChange();
    weightAdjustment();
}

// ------------------------------
// One full epoch over the training set (online learning)
void MultilayerPerceptron::trainOnline(Dataset* trainDataset)
{
    for (int p = 0; p < trainDataset->nOfPatterns; ++p)
    {
        performEpochOnline(trainDataset->inputs[p], trainDataset->outputs[p]);
    }
}

// ------------------------------
// Evaluate MSE over a dataset
double MultilayerPerceptron::test(Dataset* testDataset)
{
    double totalError = 0.0;

    for (int p = 0; p < testDataset->nOfPatterns; ++p)
    {
        feedInputs(testDataset->inputs[p]);
        forwardPropagate();
        totalError += obtainError(testDataset->outputs[p]);
    }

    return totalError / testDataset->nOfPatterns;
}

// ------------------------------
// Train for maxiter epochs and compute train/test MSE.
// Keeps the best weights according to training MSE (simple early stopping).
void MultilayerPerceptron::runOnlineBackPropagation(
    Dataset* trainDataset,
    Dataset* testDataset,
    int maxiter,
    double* errorTrain,
    double* errorTest)
{
    int epoch = 0;

    // Random assignment of weights (starting point)
    randomWeights();

    double bestTrainError = 0.0;
    int epochsWithoutImprovement = 0;

    // Learning loop
    do {
        trainOnline(trainDataset);
        double trainError = test(trainDataset);

        if (epoch == 0 || trainError < bestTrainError) {
            bestTrainError = trainError;
            copyWeights();
            epochsWithoutImprovement = 0;
        }
        else if ((trainError - bestTrainError) < 0.00001) {
            epochsWithoutImprovement = 0;
        }
        else {
            epochsWithoutImprovement++;
        }

        if (epochsWithoutImprovement == 50) {
            cout << "We exit because the training is not improving!!" << endl;
            restoreWeights();
            epoch = maxiter; // force exit
        }

        epoch++;
        cout << "Iteration " << epoch << "\t Training error: " << trainError << endl;

    } while (epoch < maxiter);

    cout << "NETWORK WEIGHTS" << endl;
    cout << "===============" << endl;
    printNetwork();

    cout << "Desired output Vs Obtained output (test)" << endl;
    cout << "=========================================" << endl;

    // Use vector to avoid manual new/delete
    std::vector<double> prediction(testDataset->nOfOutputs);

    for (int i = 0; i < testDataset->nOfPatterns; i++) {
        // Feed inputs and propagate
        feedInputs(testDataset->inputs[i]);
        forwardPropagate();
        getOutputs(prediction.data());

        for (int j = 0; j < testDataset->nOfOutputs; j++)
            cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
        cout << endl;
    }

    double testError = test(testDataset);
    *errorTest  = testError;
    *errorTrain = bestTrainError;
}


void MultilayerPerceptron::printNetwork()
{
    for (int layerIdx = 1; layerIdx < nOfLayers; ++layerIdx)
    {
        cout << "Layer " << layerIdx << endl;
        cout << "------" << endl;

        for (int neuronIdx = 0; neuronIdx < layers[layerIdx].nOfNeurons; ++neuronIdx)
        {
            for (int wIdx = 0; wIdx < layers[layerIdx - 1].nOfNeurons + 1; ++wIdx)
            {
                cout << layers[layerIdx].neurons[neuronIdx].w[wIdx] << " ";
            }
            cout << endl;
        }
    }
}


// ------------------------------
// Save the model weights into a text file
bool MultilayerPerceptron::saveWeights(const char* archivo)
{
    ofstream f(archivo);
    if (!f.is_open())
        return false;

    // Write the network topology
    f << nOfLayers;
    for (int i = 0; i < nOfLayers; i++)
        f << " " << layers[i].nOfNeurons;
    f << endl;

    // Write all weights
    for (int i = 1; i < nOfLayers; i++)
        for (int j = 0; j < layers[i].nOfNeurons; j++)
            for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
                f << layers[i].neurons[j].w[k] << " ";

    f.close();
    return true;
}
