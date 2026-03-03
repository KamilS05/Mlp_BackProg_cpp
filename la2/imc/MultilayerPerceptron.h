#ifndef _MULTILAYERPERCEPTRON_H_
#define _MULTILAYERPERCEPTRON_H_

#include "util.h"

namespace imc {

// ------------------------------
// Basic data structures
struct Neuron {
    double  out;        // Neuron output
    double  delta;      // Backpropagation delta
    double* w;          // Weight vector (bias at index 0)
    double* deltaW;     // Accumulated weight updates
    double* lastDeltaW; // Previous weight updates (momentum)
    double* wCopy;      // Copy of weights (best model)
};

struct Layer {
    int     nOfNeurons; // Number of neurons
    Neuron* neurons;    // Array of neurons
};

class MultilayerPerceptron {
private:
    int    nOfLayers;            // Total number of layers
    Layer* layers;               // Network layers
    int    nOfTrainingPatterns;  // Used in offline training

    // Memory management
    void freeMemory();

    // Core operations
    void randomWeights();
    void feedInputs(double* input);
    void getOutputs(double* output);

    // Best-weights handling
    void copyWeights();
    void restoreWeights();

    // Forward / backward propagation
    void forwardPropagate();
    double obtainError(double* target, int errorFunction);
    void backpropagateError(double* target, int errorFunction);

    // Weight updates
    void accumulateChange();
    void weightAdjustment();

    // One training step
    void performEpoch(double* input, double* target, int errorFunction);

    void printNetwork();
    void printConfusionMatrix(util::Dataset* dataset);
    void printMisclassificationsInline(util::Dataset* dataset, int nPerClass);

public:
    // Hyperparameters
    double eta;          // Learning rate
    double mu;           // Momentum factor
    bool   online;       // Online (true) or offline (false) training
    int    outputFunction; // 0 = sigmoid, 1 = softmax

    // Constructor / destructor
    MultilayerPerceptron();
    ~MultilayerPerceptron();

    // Network initialization
    int initialize(int nl, int npl[]);

    // Evaluation
    double test(util::Dataset* dataset, int errorFunction);
    double testClassification(util::Dataset* dataset);

    // Prediction (Kaggle mode)
    void predict(util::Dataset* testDataset);

    // Training
    void train(util::Dataset* trainDataset, int errorFunction);
    void runBackPropagation(util::Dataset* trainDataset,
                            util::Dataset* testDataset,
                            int maxiter,
                            double* errorTrain,
                            double* errorTest,
                            double* ccrTrain,
                            double* ccrTest,
                            int errorFunction);

    // Save / load weights
    bool saveWeights(const char* fileName);
    bool readWeights(const char* fileName);
};

} // namespace imc

#endif