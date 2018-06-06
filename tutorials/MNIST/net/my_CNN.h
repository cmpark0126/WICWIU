#include <iostream>
#include <string>

#include "../../WICWIU_src/NeuralNetwork.h"

class my_CNN : public NeuralNetwork<float>{
private:
public:
    my_CNN(Tensorholder<float> *x, Tensorholder<float> *label) {
        Operator<float> *out = NULL;

        out = AddOperator(new Reshape<float>(x, 28, 28, "reshape"));
#if __CUDNN__
        // out = AddOperator(new CUDNNBatchNormalizeLayer2D<float>(out, 1, "1"));
#endif  // __CUDNN
        // ======================= layer 1=======================
        out = AddOperator(new ConvolutionLayer2D<float>(out, 1, 10, 3, 3, 1, 1, 0, TRUE, "1"));
#if __CUDNN__
        // out = AddOperator(new CUDNNBatchNormalizeLayer2D<float>(out, 10, "1"));
#endif  // __CUDNN
        out = AddOperator(new Relu<float>(out, "Relu_1"));
        out = AddOperator(new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_1"));

        // ======================= layer 2=======================
        out = AddOperator(new ConvolutionLayer2D<float>(out, 10, 20, 3, 3, 1, 1, 0, TRUE, "2"));
#if __CUDNN__
        // out = AddOperator(new CUDNNBatchNormalizeLayer2D<float>(out, 20, "1"));
#endif  // __CUDNN
        out = AddOperator(new Relu<float>(out, "Relu_2"));
        out = AddOperator(new Maxpooling2D<float>(out, 2, 2, 2, 2, "MaxPool_2"));

        out = AddOperator(new Reshape<float>(out, 1, 1, 5 * 5 * 20, "Flat"));

        // ======================= layer 3=======================
        out = AddOperator(new Linear<float>(out, 5 * 5 * 20, 10, TRUE, "3"));

        // out = AddOperator(new Relu<float>(out, "Relu_3"));
        //
        // // ======================= layer 4=======================
        // out = AddOperator(new Linear<float>(out, 1024, 10, TRUE, "4"));

        // ======================= Select LossFunction Function ===================
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        SetOptimizer(new GradientDescentOptimizer<float>(GetTensorholder(), 0.04, MINIMIZE));
        // SetOptimizer(new GradientDescentOptimizer<float>(GetTensorholder(), 0.001, MINIMIZE));
    }

    virtual ~my_CNN() {}
};