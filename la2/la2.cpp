#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <float.h>

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
         << " -t <train.dat> [-T <test.dat>] [-i <iters>] [-l <hidden_layers>] [-h <neurons_per_hidden>]\n"
         << "        [-e <eta>] [-m <mu>] [-o] [-f <0|1>] [-s] [-n] [-w <weights.txt>] [-p]\n\n"
         << "Modes:\n"
         << "  Training/Evaluation (default): requires -t, optional -T\n"
         << "  Prediction (Kaggle): use -p -w <weights> -T <test.dat>\n\n"
         << "Arguments:\n"
         << "  -t  training dataset file (required in training mode)\n"
         << "  -T  test dataset file (optional in training mode; required in prediction mode)\n"
         << "  -i  iterations (default 1000)\n"
         << "  -l  number of hidden layers (default 1)\n"
         << "  -h  neurons per hidden layer (default 5)\n"
         << "  -e  eta learning rate (default 0.1)\n"
         << "  -m  mu momentum (default 0.9)\n"
         << "  -o  online training flag (sets mlp.online = true)\n"
         << "  -f  error function: 0=MSE, 1=Cross-Entropy (default 0)\n"
         << "  -s  softmax output flag (sets mlp.outputFunction = 1)\n"
         << "  -n  normalize inputs to [-1,1] using training stats\n"
         << "  -w  weights file (save best weights in training mode; load weights in prediction mode)\n"
         << "  -p  prediction mode (Kaggle)\n";
}

static double mean5(const double v[5])
{
    return (v[0] + v[1] + v[2] + v[3] + v[4]) / 5.0;
}

static double sd5(const double v[5], double m)
{
    double s = 0.0;
    for (int i = 0; i < 5; ++i) {
        double d = v[i] - m;
        s += d * d;
    }
    return sqrt(s / 5.0);
}

int main(int argc, char** argv)
{
    // CLI flags (keep same behavior as your original la2)
    bool Tflag = false, wflag = false, pflag = false;
    bool oflag = false;      // -o => online
    bool sflag = false;      // -s => softmax output
    bool nflag = false;      // -n => normalize inputs [-1,1]

    const char* tvalue = NULL;
    const char* Tvalue = NULL;
    const char* wvalue = NULL;

    int iterations = 1000;
    int hiddenLayers = 1;
    int neurons = 5;
    double eta = 0.1;
    double mu  = 0.9;
    int errorFunction = 0;   // 0=MSE, 1=Cross-Entropy

    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "t:T:w:pi:l:h:e:m:of:sn")) != -1)
    {
        switch (c)
        {
            case 't': tvalue = optarg; break;
            case 'T': Tflag = true; Tvalue = optarg; break;
            case 'w': wflag = true; wvalue = optarg; break;
            case 'p': pflag = true; break;
            case 'i': iterations = atoi(optarg); break;
            case 'l': hiddenLayers = atoi(optarg); break;
            case 'h': neurons = atoi(optarg); break;
            case 'e': eta = atof(optarg); break;
            case 'm': mu  = atof(optarg); break;
            case 'o': oflag = true; break;
            case 'f': errorFunction = atoi(optarg); break;
            case 's': sflag = true; break;
            case 'n': nflag = true; break;
            default:
                printUsage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // ============================
    // PREDICTION MODE (KAGGLE)
    // ============================
    if (pflag)
    {
        if (!wflag || !wvalue) {
            cerr << "Error: prediction mode (-p) requires -w <weights_file>.\n";
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
        if (!Tflag || !Tvalue) {
            cerr << "Error: prediction mode (-p) requires -T <test_file>.\n";
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }

        MultilayerPerceptron mlp;

        if (!mlp.readWeights(wvalue)) {
            cerr << "Error: could not read weights from: " << wvalue << "\n";
            return EXIT_FAILURE;
        }

        Dataset* testDataset = readData(Tvalue);
        if (!testDataset) {
            cerr << "Error: cannot read test file: " << Tvalue << "\n";
            return EXIT_FAILURE;
        }

        // Keep behavior: mlp.predict writes Kaggle output (as in your original code)
        mlp.predict(testDataset);

        freeDataset(testDataset);
        return EXIT_SUCCESS;
    }

    // ============================
    // TRAINING / EVALUATION MODE
    // ============================
    if (!tvalue)
    {
        cerr << "Error: -t <train_file> is required.\n";
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    Dataset* trainDataset = readData(tvalue);
    if (!trainDataset)
    {
        cerr << "Error: cannot read training file: " << tvalue << "\n";
        return EXIT_FAILURE;
    }

    Dataset* testDataset = trainDataset;
    bool testIsSeparate = false;

    if (Tflag)
    {
        testDataset = readData(Tvalue);
        if (!testDataset)
        {
            cerr << "Error: cannot read test file: " << Tvalue << "\n";
            freeDataset(trainDataset);
            return EXIT_FAILURE;
        }
        testIsSeparate = true;
    }

    // Optional normalization of inputs only (classification assignment behavior)
    if (nflag)
    {
        double* minInputs = minDatasetInputs(trainDataset);
        double* maxInputs = maxDatasetInputs(trainDataset);

        minMaxScalerDataSetInputs(trainDataset, -1.0, 1.0, minInputs, maxInputs);
        if (testIsSeparate)
            minMaxScalerDataSetInputs(testDataset, -1.0, 1.0, minInputs, maxInputs);

        delete[] minInputs;
        delete[] maxInputs;
    }

    // Build topology: input + hidden(s) + output
    const int nLayers = hiddenLayers + 2;
    int* topology = new int[nLayers];
    topology[0] = trainDataset->nOfInputs;
    for (int i = 1; i < nLayers - 1; ++i) topology[i] = neurons;
    topology[nLayers - 1] = trainDataset->nOfOutputs;

    // Configure MLP
    MultilayerPerceptron mlp;
    mlp.eta = eta;
    mlp.mu  = mu;
    mlp.online = oflag ? true : false;           // keep original behavior
    mlp.outputFunction = sflag ? 1 : 0;          // keep original behavior
    mlp.initialize(nLayers, topology);

    // 5 seeds (keep original behavior)
    int seeds[5] = {1, 2, 3, 4, 5};
    double trainErrors[5], testErrors[5], trainCCRs[5], testCCRs[5];
    double bestTestError = DBL_MAX;

    for (int i = 0; i < 5; ++i)
    {
        cout << "**********\n";
        cout << "SEED " << seeds[i] << "\n";
        cout << "**********\n";

        srand(seeds[i]);

        mlp.runBackPropagation(trainDataset, testDataset, iterations,
                               &trainErrors[i], &testErrors[i],
                               &trainCCRs[i], &testCCRs[i],
                               errorFunction);

        cout << "We end!! => Final test CCR: " << testCCRs[i] << endl;

        // Save best weights across seeds (same rule as your original code)
        if (wflag && wvalue && testErrors[i] <= bestTestError) {
            mlp.saveWeights(wvalue);
            bestTestError = testErrors[i];
        }
    }

    // Final report: mean +/- sd
    double trM  = mean5(trainErrors);
    double teM  = mean5(testErrors);
    double trS  = sd5(trainErrors, trM);
    double teS  = sd5(testErrors, teM);

    double trCM = mean5(trainCCRs);
    double teCM = mean5(testCCRs);
    double trCS = sd5(trainCCRs, trCM);
    double teCS = sd5(testCCRs, teCM);

    cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;
    cout << "FINAL REPORT\n*************\n";
    cout << "Train error (Mean +- SD): " << trM  << " +- " << trS  << endl;
    cout << "Test error (Mean +- SD): "  << teM  << " +- " << teS  << endl;
    cout << "Train CCR (Mean +- SD): "   << trCM << " +- " << trCS << endl;
    cout << "Test CCR (Mean +- SD): "    << teCM << " +- " << teCS << endl;

    delete[] topology;

    if (testIsSeparate) freeDataset(testDataset);
    freeDataset(trainDataset);

    return EXIT_SUCCESS;
}