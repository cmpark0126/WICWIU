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


    ListOfTensor* GetData() { // or BringData 용어변경
        ListOfTensor *temp = NULL;

        // sem_wait(sem_full)
        // sem_wait(sem_mtx)

        // Assign ListOfTensor(img Tensor, label Tensor, etc.) to temp from Buffer
        // 

        // sem_post(sem_mtx)
        // sem_post(sem_empty)

        return temp;
    }

    ListOfTensor* GetDataFromBuffer() {
        // 세마포어 확인하고 Get~ 함수 부를지 Get~ 함수에서 확인할지

        // 1안 이라면 
    }


    int GenerateData() {
        // get data from directory or user or etc
        // 

    }


    int ReadDataFromDirectory() { // 용어변경
        // go to directory 

        // read data batch size (remember iter - next data to read)

        // convert data to image (cover various data type)
    }


    ListOfImage* ConvertData2Image() { // data - image - tensor 삼단변화를 해야하나..? what to return?
        // convert various type of data to image that can perform operation or proprocessing of augmentation, etc...
    }


    ListOfTensor* ConvertImages2Tensor() { 

        // convert image to tensor(the final output)

        // rank 순서 유의하면서 바꾸어야한다,,  
    }


    ListOfImage* DataPreprocessing() {
        // pytorch 처럼 list of compose나 class 받아서 바꿀지 어떻게 할지 ㅠㅠ
    }


    ListOfImage* DataAugmentation() {
        // batch size로 받아오는데 augmentation을 하면 batch size에 안 맞게 되지 않을까..?
    }


    ListOfTensor* Concatenate() {
        // DataLoader에서만 사용하는 image라는 type이 생긴ㄱ다면 필요없어지지 않을까..?

    }


    int InsertTensor2Buffer() { // put? push? 용어변경
        //
    }

};
