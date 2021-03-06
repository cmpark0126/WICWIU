#ifndef __TRANSPOSEDCONVOLUTION_LAYER__
#define __TRANSPOSEDCONVOLUTION_LAYER__    value

#include "../Module.hpp"

template<typename DTYPE> class TransposedConvolutionLayer2D : public Module<DTYPE>{
private:
public:
    TransposedConvolutionLayer2D(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPadding, int use_bias = FALSE, std::string pName = "NO NAME") : Module<DTYPE>(pName){
        Alloc(pInput, pNumInputChannel, pNumOutputChannel, pNumKernelRow, pNumKernelCol, pStrideRow, pStrideCol, pPadding, pPadding, use_bias, pName);
    }

    virtual ~TransposedConvolutionLayer2D() {}

    int Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPaddingRow, int pPaddingCol, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        Tensorholder<DTYPE> *pWeight = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, pNumInputChannel, pNumOutputChannel, pNumKernelRow, pNumKernelCol, 0.0, 0.1), "TransposedConvolution2D_Weight_" + pName);
        out = new TransposedConvolution2D<DTYPE>(out, pWeight, pStrideRow, pStrideCol, pPaddingRow, pPaddingCol, "TransposedConvolution2D_Convolution2D_" + pName);

        if(use_bias){
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumOutputChannel, 1, 1, 0), "TransposedConvolution2D_Bias_" + pName);
            out = new AddChannelWise<DTYPE>(out, pBias, "TransposedConvolution2D_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __TRANSPOSEDCONVOLUTION_LAYER__
