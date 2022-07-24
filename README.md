# BP-MNIST-Parallel-Computing
The parallel implementation of MNIST handwritten numerals based on BP neural network
## 介绍

data:MNIST数据集
code reference：[Neural-Network-MNIST-CPP](https://github.com/HyTruongSon/Neural-Network-MNIST-CPP)

## 代码运行

[配置OpenMP和MPI](https://blog.csdn.net/qq_41315788/article/details/124088788?spm=1001.2014.3001.5501)

```c
// 串行编译
g++ -o Serial Train_Serial.cpp
// 运行
.\Serial.exe
// openmp编译
g++ -fopenmp -o OpenMP Train_Parallel_OpenMP.cpp 
// 运行
.\OpenMP
    
// MPI编译
g++ -I"D:\Program Files (x86)\Microsoft SDKs\MPI\Include" -o MPI Train_Parallel_MPI.cpp "D:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\msmpi.lib"  
// 运行
mpiexec -n 12 MPI.exe
```

## 参考博客链接

[并行化实现基于BP神经网络的手写体数字识别](https://blog.csdn.net/qq_41645895/article/details/86640526)

[BP神经网络公式推导](https://blog.csdn.net/qq_38853759/article/details/121930413)

[神经网络实现手写数字识别（MNIST）](https://blog.csdn.net/xuanwolanxue/article/details/71565934)

[OpenMP中的private/firstprivate/lastprivate/threadprivate之间的比较](https://www.cnblogs.com/DarrenChan/p/6853556.html)

[爱奇艺的一个视频关于openmp的BP神经网络实现](https://www.iqiyi.com/w_19s94ubbgt.html)