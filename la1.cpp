
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <string.h>
#include <math.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"

using namespace imc;
using namespace std;
using namespace util;



static void freeDataset(Dataset* ds)
{
    if (!ds) return;
    for (int i = 0; i < ds->nOfPatterns; ++i)
    {
        delete[] ds->inputs[i];
        delete[] ds->outputs[i];
    }
    delete[] ds->inputs;
    delete[] ds->outputs;
    delete ds;
}

static void printUsage(const char* prog)
{
    cerr << "Usage:\n  " << prog
         << " -t <train_file> [-T <test_file>] [-i <iters>] [-l <hidden_layers>] [-h <neurons_per_hidden>]\n"
         << "        [-e <eta>] [-m <mu>] [-s] [-w <weights_out>]\n\n"
         << "Notes:\n"
         << "  -t  training dataset file (required)\n"
         << "  -T  test dataset file (optional; if omitted, training data is used as test)\n"
         << "  -i  iterations (default 1000)\n"
         << "  -l  number of hidden layers (default 1)\n"
         << "  -h  neurons per hidden layer (default 5)\n"
         << "  -e  eta learning rate (default 0.1)\n"
         << "  -m  mu momentum (default 0.9)\n"
         << "  -s  normalize inputs/outputs (min-max using training set statistics)\n"
         << "  -w  save final weights to file\n";
}

int main(int argc, char **argv)
{
    // CLI args
    bool Tflag = false, wflag = false, sflag = false;
    const char *tvalue = NULL, *Tvalue = NULL, *wvalue = NULL;

    int iterations = 1000;
    int hiddenLayers = 1;
    int neurons = 5;
    double eta = 0.1;
    double mu = 0.9;

    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:sw:")) != -1)
    {
        switch (c)
        {
            case 't': tvalue = optarg; break;
            case 'T': Tflag = true; Tvalue = optarg; break;
            case 'i': iterations = atoi(optarg); break;
            case 'l': hiddenLayers = atoi(optarg); break;
            case 'h': neurons = atoi(optarg); break;
            case 'e': eta = atof(optarg); break;
            case 'm': mu = atof(optarg); break;
            case 's': sflag = true; break;
            case 'w': wflag = true; wvalue = optarg; break;
            default:
                printUsage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!tvalue)
    {
        cerr << "Error: -t <train_file> is required.\n";
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // Read datasets
    Dataset *trainDataset = readData(tvalue);
    if (!trainDataset)
    {
        cerr << "Error: cannot read training file: " << tvalue << "\n";
        return EXIT_FAILURE;
    }

    Dataset *testDataset = trainDataset;
    if (Tflag)
    {
        testDataset = readData(Tvalue);
        if (!testDataset)
        {
            cerr << "Error: cannot read test file: " << Tvalue << "\n";
            freeDataset(trainDataset);
            return EXIT_FAILURE;
        }
    }

    // Optional normalization (using TRAIN stats)
    if (sflag)
    {
        double *minInputs = minDatasetInputs(trainDataset);
        double *maxInputs = maxDatasetInputs(trainDataset);
        double minOutput = minDatasetOutputs(trainDataset);
        double maxOutput = maxDatasetOutputs(trainDataset);

        // Inputs to [-1, 1]
        minMaxScalerDataSetInputs(trainDataset, -1.0, 1.0, minInputs, maxInputs);
        if (Tflag) minMaxScalerDataSetInputs(testDataset, -1.0, 1.0, minInputs, maxInputs);

        // Outputs to [0, 1]
        minMaxScalerDataSetOutputs(trainDataset, 0.0, 1.0, minOutput, maxOutput);
        if (Tflag) minMaxScalerDataSetOutputs(testDataset, 0.0, 1.0, minOutput, maxOutput);

        delete[] minInputs;
        delete[] maxInputs;
    }

    // Build topology: input + hidden(s) + output
    const int nLayers = hiddenLayers + 2;
    int *topology = new int[nLayers];
    topology[0] = trainDataset->nOfInputs;
    for (int i = 1; i < nLayers - 1; ++i) topology[i] = neurons;
    topology[nLayers - 1] = trainDataset->nOfOutputs;

    // Initialize MLP
    MultilayerPerceptron mlp;
    mlp.eta = eta;
    mlp.mu = mu;
    mlp.initialize(nLayers, topology);

    // Deterministic run (so results are reproducible)
    srand(1);

    double trainError = 0.0, testError = 0.0;
    mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations, &trainError, &testError);

    cout << "Train error: " << trainError << "\n";
    cout << "Test  error: " << testError << "\n";

    if (wflag && wvalue)
    {
        if (!mlp.saveWeights(wvalue))
            cerr << "Warning: could not save weights to " << wvalue << "\n";
    }

    delete[] topology;

    if (Tflag) freeDataset(testDataset);
    freeDataset(trainDataset);

    return EXIT_SUCCESS;
}
