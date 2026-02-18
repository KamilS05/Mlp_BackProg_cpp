#ifndef UTIL_H_
#define UTIL_H_

#include <cstdlib> // rand(), srand()

namespace util
{
    // Structure representing a dataset
    struct Dataset
    {
        int nOfInputs;
        int nOfOutputs;
        int nOfPatterns;
        double** inputs;
        double** outputs;
    };

    // Random number utilities
    int randomInt(int Low, int High);
    double randomDouble(double Low, double High);

    // Dataset I/O and visualization
    Dataset* readData(const char* fileName);
    void printDataset(Dataset* dataset, int len);

    // Helper function to generate unique random integers (used internally)
    static int* integerRandomVectoWithoutRepeating(int min, int max, int howMany)
    {
        int total = max - min + 1;
        int* available = new int[total];
        int* selected  = new int[howMany];

        for (int i = 0; i < total; ++i)
            available[i] = min + i;

        for (int i = 0; i < howMany; ++i)
        {
            int idx = rand() % (total - i);
            selected[i] = available[idx];
            available[idx] = available[total - i - 1];
        }

        delete[] available;
        return selected;
    }

    // Min-max normalization utilities
    double minMaxScaler(double x, double minAllowed, double maxAllowed,
                        double minData, double maxData);

    void minMaxScalerDataSetInputs(Dataset* dataset,
                                   double minAllowed, double maxAllowed,
                                   double* minData, double* maxData);

    void minMaxScalerDataSetOutputs(Dataset* dataset,
                                    double minAllowed, double maxAllowed,
                                    double minData, double maxData);

    // Dataset statistics
    double* maxDatasetInputs(Dataset* dataset);
    double* minDatasetInputs(Dataset* dataset);
    double  minDatasetOutputs(Dataset* dataset);
    double  maxDatasetOutputs(Dataset* dataset);
}

#endif /* UTIL_H_ */
