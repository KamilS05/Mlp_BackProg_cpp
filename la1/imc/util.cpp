#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>   // srand(), rand()

#include "util.h"

using namespace std;
using namespace util;

// ------------------------------
// This function generates a random integer value between Low and High (inclusive).
// It is mainly used for random initialization and experimentation.
int util::randomInt(int Low, int High)
{
    return rand() % (High - Low + 1) + Low;
}

// ------------------------------
// This function generates a random floating-point value in the range [Low, High].
// It is used to initialize neural network weights.
double util::randomDouble(double Low, double High)
{
    return (static_cast<double>(rand()) / RAND_MAX) * (High - Low) + Low;
}

// ------------------------------
// This function reads a dataset from a text file and stores it in a Dataset structure.
// The first line of the file specifies the number of inputs, outputs and patterns.
// Each subsequent line contains input values followed by output values.
Dataset* util::readData(const char* fileName)
{
    ifstream myFile(fileName);
    if (!myFile.is_open())
    {
        cout << "ERROR: cannot open file " << fileName << endl;
        return NULL;
    }

    Dataset* dataset = new Dataset;
    if (!dataset) return NULL;

    string line;

    // Read dataset dimensions from the header line
    if (myFile.good())
    {
        getline(myFile, line);
        istringstream iss(line);
        iss >> dataset->nOfInputs >> dataset->nOfOutputs >> dataset->nOfPatterns;
    }

    // Allocate memory for input and output patterns
    dataset->inputs  = new double*[dataset->nOfPatterns];
    dataset->outputs = new double*[dataset->nOfPatterns];

    for (int p = 0; p < dataset->nOfPatterns; ++p)
    {
        dataset->inputs[p]  = new double[dataset->nOfInputs];
        dataset->outputs[p] = new double[dataset->nOfOutputs];
    }

    // Read each pattern line and store its values
    int p = 0;
    while (myFile.good())
    {
        getline(myFile, line);
        if (line.empty()) continue;

        istringstream iss(line);

        // Read input values
        for (int i = 0; i < dataset->nOfInputs; ++i)
        {
            double value;
            iss >> value;
            if (!iss) return NULL;
            dataset->inputs[p][i] = value;
        }

        // Read output values
        for (int o = 0; o < dataset->nOfOutputs; ++o)
        {
            double value;
            iss >> value;
            if (!iss) return NULL;
            dataset->outputs[p][o] = value;
        }

        p++;
    }

    myFile.close();
    return dataset;
}

// ------------------------------
// This function prints the contents of the dataset to standard output.
// It can print either all patterns or only the first 'len' patterns.
void util::printDataset(Dataset* dataset, int len)
{
    if (len == 0) len = dataset->nOfPatterns;

    for (int p = 0; p < len; ++p)
    {
        cout << "P" << p << ":\n";

        for (int i = 0; i < dataset->nOfInputs; ++i)
            cout << dataset->inputs[p][i] << ",";

        for (int o = 0; o < dataset->nOfOutputs; ++o)
            cout << dataset->outputs[p][o] << ",";

        cout << endl;
    }
}

// ------------------------------
// This helper function rescales a single value x into a new range
// using the original minimum and maximum values of the data.
double util::minMaxScaler(double x,
                          double minAllowed, double maxAllowed,
                          double minData, double maxData)
{
    return minAllowed + (x - minData) * (maxAllowed - minAllowed) / (maxData - minData);
}

// ------------------------------
// This function applies min-max normalization to all input features
// of the dataset using feature-wise minimum and maximum values.
void util::minMaxScalerDataSetInputs(Dataset* dataset,
                                     double minAllowed, double maxAllowed,
                                     double* minData, double* maxData)
{
    for (int p = 0; p < dataset->nOfPatterns; ++p)
    {
        for (int i = 0; i < dataset->nOfInputs; ++i)
        {
            dataset->inputs[p][i] = minMaxScaler(dataset->inputs[p][i],
                                                 minAllowed, maxAllowed,
                                                 minData[i], maxData[i]);
        }
    }
}

// ------------------------------
// This function applies min-max normalization to the output values
// of the dataset. It is mainly intended for regression problems.
void util::minMaxScalerDataSetOutputs(Dataset* dataset,
                                      double minAllowed, double maxAllowed,
                                      double minData, double maxData)
{
    for (int p = 0; p < dataset->nOfPatterns; ++p)
    {
        for (int o = 0; o < dataset->nOfOutputs; ++o)
        {
            dataset->outputs[p][o] = minMaxScaler(dataset->outputs[p][o],
                                                  minAllowed, maxAllowed,
                                                  minData, maxData);
        }
    }
}

// ------------------------------
// This function computes the minimum value for each input feature
// across the entire dataset.
double* util::minDatasetInputs(Dataset* dataset)
{
    double* minInputs = new double[dataset->nOfInputs];

    for (int i = 0; i < dataset->nOfInputs; ++i)
    {
        minInputs[i] = dataset->inputs[0][i];
        for (int p = 1; p < dataset->nOfPatterns; ++p)
        {
            if (dataset->inputs[p][i] < minInputs[i])
                minInputs[i] = dataset->inputs[p][i];
        }
    }
    return minInputs;
}

// ------------------------------
// This function computes the maximum value for each input feature
// across the entire dataset.
double* util::maxDatasetInputs(Dataset* dataset)
{
    double* maxInputs = new double[dataset->nOfInputs];

    for (int i = 0; i < dataset->nOfInputs; ++i)
    {
        maxInputs[i] = dataset->inputs[0][i];
        for (int p = 1; p < dataset->nOfPatterns; ++p)
        {
            if (dataset->inputs[p][i] > maxInputs[i])
                maxInputs[i] = dataset->inputs[p][i];
        }
    }
    return maxInputs;
}

// ------------------------------
// This function finds the smallest output value present in the dataset.
double util::minDatasetOutputs(Dataset* dataset)
{
    double minOutput = dataset->outputs[0][0];

    for (int p = 0; p < dataset->nOfPatterns; ++p)
    {
        for (int o = 0; o < dataset->nOfOutputs; ++o)
        {
            if (dataset->outputs[p][o] < minOutput)
                minOutput = dataset->outputs[p][o];
        }
    }
    return minOutput;
}

// ------------------------------
// This function finds the largest output value present in the dataset.
double util::maxDatasetOutputs(Dataset* dataset)
{
    double maxOutput = dataset->outputs[0][0];

    for (int p = 0; p < dataset->nOfPatterns; ++p)
    {
        for (int o = 0; o < dataset->nOfOutputs; ++o)
        {
            if (dataset->outputs[p][o] > maxOutput)
                maxOutput = dataset->outputs[p][o];
        }
    }
    return maxOutput;
}
