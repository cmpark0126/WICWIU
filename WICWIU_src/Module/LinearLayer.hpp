#ifndef __LINEAR_LAYER__
#define __LINEAR_LAYER__    value

#include "../Module.hpp"

template<typename DTYPE> class Linear : public Module<DTYPE>{
private:
public:
    Linear(Operator<DTYPE> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias = FALSE, std::string pName = NULL) : Module<DTYPE>(pName) {
        Alloc(pInput, pNumInputCol, pNumOutputCol, use_bias, pName);
    }

    virtual ~Linear() {}

    int Alloc(Operator<DTYPE> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        // for He initialization
        float stddev = sqrt((float)4/(pNumInputCol + pNumOutputCol));
        // float stddev = 0.1;

        Tensorholder<DTYPE> *pWeight = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, pNumOutputCol, pNumInputCol, 0.0, stddev), "Layer_Weight_" + pName);
        out = new MatMul<DTYPE>(pWeight, out, "Layer_MatMul_" + pName);

        if (use_bias) {
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumOutputCol, 0.f), "Add_Bias_" + pName);
            out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif  // __LINEAR_LAYER__
