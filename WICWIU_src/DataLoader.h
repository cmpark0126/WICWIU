#include "NeuralNetwork.h"

class ListOfTensor {
private:
    /* data */

public:
    ListOfTensor();
    virtual ~ListOfTensor();
};

template<typename DTYPE> class DataLoader {
private:
public:
    DataLoader() {
        Alloc();
    }

    virtual ~DataLoader() {
        Delete();
    }

    int           Alloc();
    void          Delete();

    // Layer 1
    ListOfTensor* LoadData() {
        ListOfTensor *temp = NULL;

        // sem_wait(sem_full)
        // sem_wait(sem_mtx)

        // Assign ListOfTensor(img Tensor, label Tensor, etc.) to temp from Buffer

        // sem_post(sem_mtx)
        // sem_post(sem_empty)

        return temp;
    }

    int GenerateData();
};
