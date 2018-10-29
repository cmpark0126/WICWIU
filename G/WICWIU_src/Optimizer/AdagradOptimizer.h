#ifndef ADAGRADOPTIMIZER_H_
#define ADAGRADOPTIMIZER_H_   value

#include "../Optimizer.h"


template<typename DTYPE> class AdagradOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter;
    Container<Tensor<DTYPE> *> *m_aaGradientSquared;

    int m_numOfParameter;
    float m_epsilon;


public:
    AdagradOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float epsilon, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
    #ifdef __DEBUG__
    std::cout << "AdagradOptimizer::AdagradOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
    #endif  // __DEBUG__
    m_ppParameter          = NULL;
    m_aaGradientSquared      = NULL;

    m_numOfParameter       = 0;
    m_epsilon              = 0.f;

      Alloc(epsilon);
  }

    AdagradOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float epsilon, float weightDecayRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, weightDecayRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "AdagradOptimizer::AdagradOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter          = NULL;
        m_aaGradientSquared    = NULL;

        m_numOfParameter       = 0;
        m_epsilon              = 0.f;

        Alloc(epsilon);
    }

    ~AdagradOptimizer() {
        #ifdef __DEBUG__
        std::cout << "AdagradOptimizer::~AdagradOptimizer()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    virtual void Delete(){
      if (m_aaGradientSquared) {
          delete m_aaGradientSquared;
          m_aaGradientSquared = NULL;
      }
    }

    int Alloc(float epsilon){
      m_ppParameter    = this->GetTrainableTensor();
      m_numOfParameter = this->GetTrainableTensorDegree();
      m_aaGradientSquared = new Container<Tensor<DTYPE> *>();

      Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();
            m_aaGradientSquared->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            pParameterGradShape = NULL;
        }

        m_epsilon = epsilon;

        return TRUE;
    }

    virtual int UpdateParameter() {
      if(m_epsilon != 0.f){
          for (int i = 0; i < m_numOfParameter; i++) {
              UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaGradientSquared)[i]);
          }
      }else{std::cout << "Don't execute UpdateParameter"<<std::endl;}
      return TRUE;
    }

    int UpdateParameter(Operator<DTYPE> *pParameter){
      return TRUE;
    }

    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pGradientSquared) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*m_pGradientSquared)[i] = ((*gradient)[i] * (*gradient)[i]);
            (*trainable_data)[i]    += (signed_learning_rate * (*gradient)[i]) / std::sqrt((*m_pGradientSquared)[i] + m_epsilon);
        }

        return TRUE;
    }


#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        if(m_epsilon != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                (*m_aaGradientSquared)[i]->SetDeviceGPU(idOfDevice);
            }
        }else{std::cout << "Don't execute SetDeviceGPU"<< std::endl;}
    }

    virtual int UpdateParameterOnGPU() {
        if(m_epsilon != 0.f){
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaGradientSquared)[i]);
            }
        }else{std::cout << "Don't execute UpdateParameterOnGPU"<<std::endl;}
        return TRUE;
    }


    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter){
      return TRUE;
    }

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pGradientSquared);


 #endif  // if __CUDNN__
};

#endif  // ADAGRADOPTIMIZER_H_
