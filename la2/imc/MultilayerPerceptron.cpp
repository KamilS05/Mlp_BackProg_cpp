#include "MultilayerPerceptron.h"
#include "util.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <limits>
#include <math.h>
#include <float.h>
#include <vector>



using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
    nOfLayers = 0;
    layers = NULL;

    // Defaults (may be overwritten from outside)
    eta = 0.1;
    mu = 0.9;
    online = false;        // LA2: domyślnie off-line
    outputFunction = 0;    // 0=sigmoid, 1=softmax
    nOfTrainingPatterns = 0;
}

// ------------------------------
// Allocate memory for the data structures
int MultilayerPerceptron::initialize(int nl, int npl[])
{
    freeMemory();

    nOfLayers = nl;
    layers = new Layer[nOfLayers];

    for (int h = 0; h < nOfLayers; h++)
    {
        layers[h].nOfNeurons = npl[h];
        layers[h].neurons = new Neuron[layers[h].nOfNeurons];

        if (h > 0)
        {
            for (int j = 0; j < layers[h].nOfNeurons; j++)
            {
                int nInputs = layers[h-1].nOfNeurons + 1; // +1 bias
                layers[h].neurons[j].w          = new double[nInputs];
                layers[h].neurons[j].deltaW     = new double[nInputs];
                layers[h].neurons[j].lastDeltaW = new double[nInputs];
                layers[h].neurons[j].wCopy      = new double[nInputs];
                layers[h].neurons[j].out        = 0.0;
                layers[h].neurons[j].delta      = 0.0;
                for (int k = 0; k < nInputs; k++)
                {
                    layers[h].neurons[j].w[k] = 0.0;
                    layers[h].neurons[j].deltaW[k] = 0.0;
                    layers[h].neurons[j].lastDeltaW[k] = 0.0;
                    layers[h].neurons[j].wCopy[k] = 0.0;
                }
            }
        }
        else
        {
            // warstwa wejściowa: tylko out
            for (int j = 0; j < layers[h].nOfNeurons; j++)
            {
                layers[h].neurons[j].out = 0.0;
                layers[h].neurons[j].delta = 0.0;
                layers[h].neurons[j].w = NULL;
                layers[h].neurons[j].deltaW = NULL;
                layers[h].neurons[j].lastDeltaW = NULL;
                layers[h].neurons[j].wCopy = NULL;
            }
        }
    }
    return 1;
}

// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron()
{
    freeMemory();
}

// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory()
{
    if (!layers) return;
    for (int h = 0; h < nOfLayers; h++)
    {
        if (layers[h].neurons)
        {
            if (h > 0)
            {
                for (int j = 0; j < layers[h].nOfNeurons; j++)
                {
                    delete[] layers[h].neurons[j].w;
                    delete[] layers[h].neurons[j].deltaW;
                    delete[] layers[h].neurons[j].lastDeltaW;
                    delete[] layers[h].neurons[j].wCopy;
                }
            }
            delete[] layers[h].neurons;
        }
    }
    delete[] layers;
    layers = NULL;
    nOfLayers = 0;
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights()
{
    for (int h = 1; h < nOfLayers; h++)
        for (int j = 0; j < layers[h].nOfNeurons; j++)
        {
            int nInputs = layers[h-1].nOfNeurons + 1;
            for (int k = 0; k < nInputs; k++)
                layers[h].neurons[j].w[k] = randomDouble(-1.0, 1.0);
        }
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input)
{
    for (int i = 0; i < layers[0].nOfNeurons; i++)
        layers[0].neurons[i].out = input[i];
}

// ------------------------------
// Get the outputs predicted by the network
void MultilayerPerceptron::getOutputs(double* output)
{
    int L = nOfLayers - 1;
    for (int j = 0; j < layers[L].nOfNeurons; j++)
        output[j] = layers[L].neurons[j].out;
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights()
{
    for (int h = 1; h < nOfLayers; h++)
        for (int j = 0; j < layers[h].nOfNeurons; j++)
        {
            int nInputs = layers[h-1].nOfNeurons + 1;
            for (int k = 0; k < nInputs; k++)
                layers[h].neurons[j].wCopy[k] = layers[h].neurons[j].w[k];
        }
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights()
{
    for (int h = 1; h < nOfLayers; h++)
        for (int j = 0; j < layers[h].nOfNeurons; j++)
        {
            int nInputs = layers[h-1].nOfNeurons + 1;
            for (int k = 0; k < nInputs; k++)
                layers[h].neurons[j].w[k] = layers[h].neurons[j].wCopy[k];
        }
}

// ------------------------------
// Calculate and propagate the outputs of the neurons
void MultilayerPerceptron::forwardPropagate()
{
    // hidden layers: sigmoid
    for (int h = 1; h < nOfLayers; h++)
    {
        bool lastLayer = (h == nOfLayers - 1);
        if (!lastLayer)
        {
            for (int j = 0; j < layers[h].nOfNeurons; j++)
            {
                double net = layers[h].neurons[j].w[0]; // bias
                for (int i = 0; i < layers[h-1].nOfNeurons; i++)
                    net += layers[h].neurons[j].w[i+1] * layers[h-1].neurons[i].out;

                layers[h].neurons[j].out = 1.0 / (1.0 + exp(-net));
            }
        }
        else
        {
            // output layer: sigmoid or softmax
            int nOut = layers[h].nOfNeurons;
            vector<double> net(nOut);
            double maxNet = -DBL_MAX;

            for (int j = 0; j < nOut; j++)
            {
                double v = layers[h].neurons[j].w[0];
                for (int i = 0; i < layers[h-1].nOfNeurons; i++)
                    v += layers[h].neurons[j].w[i+1] * layers[h-1].neurons[i].out;
                net[j] = v;
                if (v > maxNet) maxNet = v;
            }

            if (outputFunction == 1) // softmax (safe)
            {
                double sumExp = 0.0;
                for (int j = 0; j < nOut; j++)
                    sumExp += exp(net[j] - maxNet); // stabilizacja
                for (int j = 0; j < nOut; j++)
                    layers[h].neurons[j].out = exp(net[j] - maxNet) / sumExp;
            }
            else
            {
                // sigmoid
                for (int j = 0; j < nOut; j++)
                    layers[h].neurons[j].out = 1.0 / (1.0 + exp(-net[j]));
            }
        }
    }
}

// ------------------------------
// Obtain the output error (MSE or Cross Entropy)
double MultilayerPerceptron::obtainError(double* target, int errorFunction)
{
    int L = nOfLayers - 1;
    int K = layers[L].nOfNeurons;

    const double eps = 1e-15;
    double acc = 0.0;

    if (errorFunction == 0) // MSE
    {
        for (int o = 0; o < K; o++)
        {
            double diff = target[o] - layers[L].neurons[o].out;
            acc += diff * diff;
        }
        return acc / K;
    }
    else // Cross Entropy
    {
        // L = -(1/K) * sum_o d_o * ln(o_o)
        for (int o = 0; o < K; o++)
        {
            double oo = layers[L].neurons[o].out;
            if (oo < eps) oo = eps;
            acc += target[o] * log(oo);
        }
        return -(acc / K);
    }
}

// ------------------------------
// Backpropagate the output error
void MultilayerPerceptron::backpropagateError(double* target, int errorFunction)
{
    int L = nOfLayers - 1;
    int K = layers[L].nOfNeurons;

    // Output layer deltas (4 cases)
    if (outputFunction == 0) // sigmoid in output
    {
        for (int j = 0; j < K; j++)
        {
            double out = layers[L].neurons[j].out;
            if (errorFunction == 0) // MSE
            {
                // δ = -(d - out) * out * (1-out)
                layers[L].neurons[j].delta = -(target[j] - out) * out * (1.0 - out);
            }
            else // Cross Entropy
            {
                // δ = -(d/out) * out * (1-out)   (wg slajdów LA2)
                // (równoważne out - d, ale trzymamy się instrukcji)
                double oo = out;
                if (oo < 1e-15) oo = 1e-15;
                layers[L].neurons[j].delta = -(target[j] / oo) * out * (1.0 - out);
            }
        }
    }
    else // softmax in output
    {
        if (errorFunction == 0) // MSE + softmax
        {
            // δ_j = - sum_i ( (d_i - out_i) * out_j * (I(i=j) - out_i) )
            for (int j = 0; j < K; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < K; i++)
                {
                    double Ii = (i == j) ? 1.0 : 0.0;
                    sum += ( (target[i] - layers[L].neurons[i].out) * 
                             layers[L].neurons[j].out * (Ii - layers[L].neurons[i].out) );
                }
                layers[L].neurons[j].delta = -sum;
            }
        }
        else // Cross Entropy + softmax
        {
            // δ_j = - sum_i ( (d_i / out_i) * out_j * (I(i=j) - out_i) )
            // (upraszcza się do out_j - d_j; zostawiamy formę z instrukcji)
            for (int j = 0; j < K; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < K; i++)
                {
                    double oi = layers[L].neurons[i].out;
                    if (oi < 1e-15) oi = 1e-15;
                    double Ii = (i == j) ? 1.0 : 0.0;
                    sum += ( (target[i] / oi) * layers[L].neurons[j].out * (Ii - layers[L].neurons[i].out) );
                }
                layers[L].neurons[j].delta = -sum;
            }
        }
    }

    // Hidden layers: standard backprop (sigmoid neurons)
    for (int h = L - 1; h >= 1; h--)
    {
        for (int j = 0; j < layers[h].nOfNeurons; j++)
        {
            double s = 0.0;
            for (int i = 0; i < layers[h+1].nOfNeurons; i++)
                s += layers[h+1].neurons[i].w[j+1] * layers[h+1].neurons[i].delta;

            double out = layers[h].neurons[j].out;
            layers[h].neurons[j].delta = s * out * (1.0 - out);
        }
    }
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange()
{
    for (int h = 1; h < nOfLayers; h++)
        for (int j = 0; j < layers[h].nOfNeurons; j++)
        {
            // bias
            layers[h].neurons[j].deltaW[0] += layers[h].neurons[j].delta * 1.0;
            // rest
            for (int i = 0; i < layers[h-1].nOfNeurons; i++)
                layers[h].neurons[j].deltaW[i+1] += layers[h].neurons[j].delta * layers[h-1].neurons[i].out;
        }
    
}

// ------------------------------
// Update the network weights
void MultilayerPerceptron::weightAdjustment()
{
    // For off-line mode we average derivatives over N, as in slides. :contentReference[oaicite:2]{index=2}
    double denom = (online ? 1.0 : (nOfTrainingPatterns > 0 ? (double)nOfTrainingPatterns : 1.0));

    for (int h = 1; h < nOfLayers; h++)
        for (int j = 0; j < layers[h].nOfNeurons; j++)
        {
            int nInputs = layers[h-1].nOfNeurons + 1;
            for (int i = 0; i < nInputs; i++)
            {
                double grad = layers[h].neurons[j].deltaW[i] / denom;
                double prev = layers[h].neurons[j].lastDeltaW[i] / denom;

                double change = eta * grad + mu * eta * prev;
                layers[h].neurons[j].w[i] -= change;

                layers[h].neurons[j].lastDeltaW[i] = layers[h].neurons[j].deltaW[i];
                layers[h].neurons[j].deltaW[i] = 0.0; // clear accumulator
            }
        }
}

// ------------------------------
// Print the network
void MultilayerPerceptron::printNetwork()
{
    for (int h = 1; h < nOfLayers; h++)
    {
        cout << "Layer " << h << endl;
        cout << "------" << endl;
        for (int j = 0; j < layers[h].nOfNeurons; j++)
        {
            for (int i = 0; i < layers[h-1].nOfNeurons + 1; i++)
                if(!(outputFunction==1 && (h==(nOfLayers-1)) && (j==(layers[h].nOfNeurons-1))))
                    cout << layers[h].neurons[j].w[i] << " ";
            cout << endl;
        }
    }
}

// ------------------------------
// Perform an epoch over a single pattern
void MultilayerPerceptron::performEpoch(double* input, double* target, int errorFunction)
{
    feedInputs(input);
    forwardPropagate();
    backpropagateError(target, errorFunction);
    accumulateChange();
    if (online) weightAdjustment();
}

// ------------------------------
// Train the network for a dataset (one outer-loop iteration)
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction)
{
    for (int p = 0; p < trainDataset->nOfPatterns; p++)
        performEpoch(trainDataset->inputs[p], trainDataset->outputs[p], errorFunction);

    if (!online) weightAdjustment();
}

// ------------------------------
// Test the network with a dataset and return the error
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction)
{
    double total = 0.0;
    for (int p = 0; p < dataset->nOfPatterns; p++)
    {
        feedInputs(dataset->inputs[p]);
        forwardPropagate();
        total += obtainError(dataset->outputs[p], errorFunction);
    }
    return total / dataset->nOfPatterns;
}

// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset)
{
    int L = nOfLayers - 1;
    int correct = 0;

    for (int p = 0; p < dataset->nOfPatterns; p++)
    {
        feedInputs(dataset->inputs[p]);
        forwardPropagate();

        int argmaxOut = 0, argmaxTarget = 0;
        for (int j = 1; j < layers[L].nOfNeurons; j++)
        {
            if (layers[L].neurons[j].out > layers[L].neurons[argmaxOut].out) argmaxOut = j;
            if (dataset->outputs[p][j] > dataset->outputs[p][argmaxTarget]) argmaxTarget = j;
        }
        if (argmaxOut == argmaxTarget) correct++;
    }
    return 100.0 * (double)correct / (double)dataset->nOfPatterns;
}

// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset (unchanged)
void MultilayerPerceptron::predict(Dataset* dataset)
{
    int numSalidas = layers[nOfLayers-1].nOfNeurons;
    double * salidas = new double[numSalidas];

    cout << "Id,Category" << endl;

    for (int i=0; i<dataset->nOfPatterns; i++){
        feedInputs(dataset->inputs[i]);
        forwardPropagate();
        getOutputs(salidas);

        int maxIndex = 0;
        for (int j = 0; j < numSalidas; j++)
            if (salidas[j] >= salidas[maxIndex])
                maxIndex = j;

        cout << i << "," << maxIndex << endl;
    }
    delete[] salidas;
}

// ------------------------------
// Run the training algorithm (outer loop)
void MultilayerPerceptron::runBackPropagation(Dataset * trainDataset, Dataset * testDataset,
                                              int maxiter, double *errorTrain, double *errorTest,
                                              double *ccrTrain, double *ccrTest, int errorFunction)
{
    int countTrain = 0;

    randomWeights();
    copyWeights(); // initial copy

    double minTrainError = 0.0;
    int iterWithoutImproving = 0;
    nOfTrainingPatterns = trainDataset->nOfPatterns;

    do {
        train(trainDataset, errorFunction);

        double trainErr = test(trainDataset, errorFunction);

        if(countTrain == 0 || trainErr < minTrainError - 1e-5)
        {
            minTrainError = trainErr;
            copyWeights();
            iterWithoutImproving = 0;
        }
        else
        {
            iterWithoutImproving++;
        }

        if(iterWithoutImproving == 50){
            cout << "We exit because the training is not improving!!" << endl;
            restoreWeights();
            countTrain = maxiter;
        }

        countTrain++;
        cout << "Iteration " << countTrain << "\t Training error: " << trainErr << endl;

    } while (countTrain < maxiter);

    cout << "NETWORK WEIGHTS" << endl;
    cout << "===============" << endl;
    printNetwork();

    cout << "Desired output Vs Obtained output (test)" << endl;
    cout << "=========================================" << endl;
    for(int i=0; i<testDataset->nOfPatterns; i++){
        double* prediction = new double[testDataset->nOfOutputs];
        feedInputs(testDataset->inputs[i]);
        forwardPropagate();
        getOutputs(prediction);
        for(int j=0; j<testDataset->nOfOutputs; j++)
            cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
        cout << endl;
        delete[] prediction;
    }

    *errorTrain = minTrainError;
    *errorTest  = test(testDataset, errorFunction);
    *ccrTrain   = testClassification(trainDataset);
    *ccrTest    = testClassification(testDataset);

    cout << "\nCONFUSION MATRIX (TEST)\n=======================\n";
    printConfusionMatrix(testDataset);
    printMisclassificationsInline(testDataset, 3);

}

// -------------------------
// Kaggle: Save the model weights in a textfile (kept as in skeleton)
bool MultilayerPerceptron::saveWeights(const char * fileName)
{
    ofstream f(fileName);
    if(!f.is_open())
        return false;

    f << nOfLayers;
    for(int i = 0; i < nOfLayers; i++)
        f << " " << layers[i].nOfNeurons;
    f << " " << outputFunction;
    f << endl;

    for(int i = 1; i < nOfLayers; i++)
        for(int j = 0; j < layers[i].nOfNeurons; j++)
            for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
                if(layers[i].neurons[j].w!=NULL)
                    f << layers[i].neurons[j].w[k] << " ";

    f.close();
    return true;
}

// -----------------------
// Kaggle: Load the model weights (kept as in skeleton)
bool MultilayerPerceptron::readWeights(const char * fileName)
{
    ifstream f(fileName);
    if(!f.is_open())
        return false;

    int nl; f >> nl;
    int *npl = new int[nl];
    for(int i = 0; i < nl; i++) f >> npl[i];
    f >> outputFunction;

    initialize(nl, npl);

    for(int i = 1; i < nOfLayers; i++)
        for(int j = 0; j < layers[i].nOfNeurons; j++)
            for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
                if(!(outputFunction==1 && (i==(nOfLayers-1)) && (j==(layers[i].nOfNeurons-1))))
                    f >> layers[i].neurons[j].w[k];

    f.close();
    delete[] npl;
    return true;
}

void MultilayerPerceptron::printConfusionMatrix(Dataset* dataset)
{
    int L = nOfLayers - 1;
    int K = layers[L].nOfNeurons;

    // macierz KxK wypełniona zerami
    std::vector<std::vector<int>> cm(K, std::vector<int>(K, 0));

    for (int p = 0; p < dataset->nOfPatterns; ++p) {
        // forward dla wzorca
        feedInputs(dataset->inputs[p]);
        forwardPropagate();

        // klasa prawdziwa = argmax targetu
        int trueC = 0;
        for (int j = 1; j < K; ++j)
            if (dataset->outputs[p][j] > dataset->outputs[p][trueC]) trueC = j;

        // klasa przewidziana = argmax wyjścia
        int predC = 0;
        for (int j = 1; j < K; ++j)
            if (layers[L].neurons[j].out > layers[L].neurons[predC].out) predC = j;

        cm[trueC][predC]++;
    }

    // wydruk
    cout << "\nConfusion Matrix (rows=true, cols=predicted)\n";
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            cout << cm[i][j];
            if (j < K-1) cout << " ";
        }
        cout << endl;
    }
}

static inline char letterFromIdx(int k) { return char('A' + k); }

void MultilayerPerceptron::printMisclassificationsInline(Dataset* dataset, int nPerClass)
{
    if(nPerClass <= 0) return;

    const int L = nOfLayers - 1;
    const int K = layers[L].nOfNeurons;

    std::vector<int> shownPerTrue(K, 0);

    cout << "\nMISCLASSIFIED EXAMPLES (up to " << nPerClass
         << " per true class)\n===========================================\n";
    cout << "idx\ttrue\tpred\toutputs[0.." << (K-1) << "]\n";

    int printedTotal = 0;

    for (int p = 0; p < dataset->nOfPatterns; ++p) {
        // forward
        feedInputs(dataset->inputs[p]);
        forwardPropagate();

        // argmax: true, pred
        int trueC = 0, predC = 0;
        for (int j = 1; j < K; ++j) {
            if (dataset->outputs[p][j] > dataset->outputs[p][trueC]) trueC = j;
            if (layers[L].neurons[j].out  > layers[L].neurons[predC].out) predC = j;
        }
        if (trueC == predC) continue;

        if (shownPerTrue[trueC] < nPerClass) {
            shownPerTrue[trueC]++;
            printedTotal++;

            cout << p << "\t"
                 << letterFromIdx(trueC) << "(" << trueC << ")\t"
                 << letterFromIdx(predC) << "(" << predC << ")\t";

            cout << fixed << setprecision(3);
            for (int j = 0; j < K; ++j) {
                cout << layers[L].neurons[j].out;
                if (j+1 < K) cout << ",";
            }
            cout << "\n";
        }

        // jeżeli już mamy komplet dla każdej klasy, możemy przerwać
        bool done = true;
        for (int c = 0; c < K; ++c) if (shownPerTrue[c] < nPerClass) { done = false; break; }
        if (done) break;
    }

    if (printedTotal == 0) {
        cout << "(no misclassifications found under current settings)\n";
    }
}