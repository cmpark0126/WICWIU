#ifdef __CUDNN__

#ifndef __PARALLEL_H_
#define __PARALLEL_H_    value

#include "NeuralNetwork.h"

enum GatherMode{
  BroadCast,
  AllReduceRing
}

template<typename DTYPE> class Parallel{
private:
  int * m_aSetOfDevice;
  int m_outputDevice;
  GatherMode m_mode

private:
    int  Alloc();
    void Delete();

public:
    Parallel(NeuralNetwork<DTYPE> * net, int numberOfDevice, ...);
    virtual ~Parallel();

    int SetOutputDevice(); // 설정한 디바이스 안에 있어야 한다.

    int ReplicateNetwork(); // 각 Device에 모델 복사

    int Scatter();

    int Gather(); // 분산된 결과를 하나로 모으는 역할

    int ParallelApply();

};



#endif  // ifndef __PARALLEL_H_

#endif  // if __CUDNN__
