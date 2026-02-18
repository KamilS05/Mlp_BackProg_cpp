#ifndef _MULTILAYERPERCEPTRON_H_
#define _MULTILAYERPERCEPTRON_H_

#include "util.h"

namespace imc {

// ------------------------------
// Basic data structures used by the network
struct Neuron {
    double  out;        // Output produced by the neuron
    double  delta;      // Delta value used in backpropagation
    double* w;          // Weight vector (including bias at index 0)
    double* deltaW;     // Accumulated weight changes for the current update
    double* lastDeltaW; // Previous weight changes (momentum term)
    double* wCopy;      // Copy of weights used to restore best model
};

struct Layer {
    int     nOfNeurons; // Number of neurons in this layer
    Neuron* neurons;    // Array of neurons
};

class MultilayerPerceptron {
private:
    int    nOfLayers;   // Total number of layers (input + hidden + output)
    Layer* layers;      // Array of layers

    // Memory management for internal structures
    void freeMemory();

    // Core network operations
    void randomWeights();
    void feedInputs(double* input);
    void getOutputs(double* output);

    // Best-weights bookkeeping (simple early stopping)
    void copyWeights();
    void restoreWeights();

    // Forward / backward passes and weight updates
    void forwardPropagate();
    double obtainError(double* target);
    void backpropagateError(double* target);
    void accumulateChange();
    void weightAdjustment();

    // One online SGD step for a single training pattern
    void performEpochOnline(double* input, double* target);
	void printNetwork();
	
public:
    // Hyperparameters (can be set from outside)
    double eta; // Learning rate
    double mu;  // Momentum factor

    // Constructor / destructor
    MultilayerPerceptron();
    ~MultilayerPerceptron();

    // Allocate memory for a network with nl layers and npl neurons per layer
    int initialize(int nl, int npl[]);

    // Train for one epoch over the dataset (online learning)
    void trainOnline(util::Dataset* trainDataset);

    // Evaluate MSE over a dataset
    double test(util::Dataset* dataset);

    // Train for maxiter epochs and return train/test errors
    void runOnlineBackPropagation(util::Dataset* trainDataset,
                                  util::Dataset* testDataset,
                                  int maxiter,
                                  double* errorTrain,
                                  double* errorTest);

    // Save model weights to a text file
    bool saveWeights(const char* archivo);
};

} // namespace imc

#endif
