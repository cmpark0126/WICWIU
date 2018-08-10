#ifdef __CUDNN__

#ifndef __PARALLEL_H_
#define __PARALLEL_H_    value

#include "NeuralNetwork.h"

#ifndef SyncBetweenDevices
  #define  BroadCast     0
  #define  AllReduceRing 1
#endif

#ifndef ThreadOrProcess
  #define  Thread     0
  #define  Process    1
#endif

template<typename DTYPE> class DataParallel{
private:
  int * m_aSetOfDevice;
  int m_numOfDevice;
  int m_outputDevice;

  Container<Operator<DTYPE> *> * m_aaInputs;
  Container<NeuralNetwork<DTYPE> *> *m_aaDistributedNets;
  Container<Operator<DTYPE> *> *m_apParameter;

private:
  int  Alloc(){
    m_aSetOfDevice = new int(numberOfDevice);
    m_numOfDevice = numberOfDevice;

    for (int i = 0; i < numberOfDevice){
      // m_aSetOfDevice[i] = value;

    }

    return TRUE;
  }
  void Delete(){
    if(m_aSetOfDevice) {
      delete m_aSetOfDevice;
      m_aSetOfDevice = NULL;
    }

    if(m_aaInputs){
      for(int i = 0; i < size; i++){
        delete m_aaInputs[i];
        m_aaInputs[i] = NULL;
      }
      delete m_aaInputs;
      m_aaInputs = NULL;
    }

  }

public:
  Parallel(NeuralNetwork<DTYPE> * net,  int numberOfDevice, ...){
    // 가변함수 이용하는 코드 짜기
    // config file을 만들지 생각중

    Alloc();
  }

  virtual ~Parallel();

  int SetOutputDevice(); // 설정한 디바이스 안에 있어야 한다.

  int Replicate(){ // 각 Device에 모델 복사
    // multi thread로 구현
    // 각 네트워크를 정의할 때, 공간과 설계도만 구현하고, 할당은 기존의 네트워크가 해제된 이후에 진행하는 것으로 한다.

    for(int i = 0; i < m_numOfDevice; i++){
      // 이 알고리즘을 위해서 nchw를 사용한 batch 조절이 필요, Copy 알고리즘 구현, 그래프의 Edge 구조를 유지시킨 것을 만든다. Deep copy와 비슷하지만, 동적할당을 미룬다는 점에서 차이를 둔다.
      // m_aaDistributedModel->Push(net->Copy(noAlloc));
      // (*m_aaDistributedModel)[i]->SetBatchSize();
    }

    // 기존 네트워크 Delete - 이쪽에서 요소들을 꺼내서 하나하나 해제 시켜주어야 함.

    for(int i = 0; i < m_numOfDevice; i++){
      // (*m_aaDistributedModel)[i]->Alloc();
      // (*m_aaDistributedModel)[i]->SetDeviceGPU(m_aSetOfDevice[i]);
    }

  }

  int Scatter(); // 첫 번째 레이어의 입력을 분산 시킴

  int Gather(); // 분산된 결과를 하나로 모으는 역할

  int ParallelApply();

  int SetAlgorithm();

  int SetMultiThread();

  int SetMultiProcess();

  int ForwardPropagateOnGPU();

  int BackPropagateOnGPU();

  int Training();

  int Testing();

};



#endif  // ifndef __PARALLEL_H_

#endif  // if __CUDNN__
