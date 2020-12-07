## Inertial-Gesture-Recognition

[https://github.com/bczhangbczhang/Inertial-Gesture-Recognition](https://github.com/bczhangbczhang/Inertial-Gesture-Recognition)

________________
### Introduction
Mobile phone plays an important role in our daily life. This paper develops a gesture recognition benchmark based on sensors of mobile phone. The built-in micro gyroscope and accelerometer of mobile phone can efficiently measure the accelerations and angular velocities along x-,y- and z-axis, which are used as the input data. We calculate the energy of the input data to reduce the effect of the phone’s posture variations. A large database is collected, which contains more than 1,000 samples of 8 gestures. The Hidden Markov Model (HMM), K-NearestNeighbor (KNN) and Support Vector Machine (SVM) are tested on the benchmark. The experimental results indicated that the employed methods can effectively recognize the gestures. To promote research on this topic, the source code and database are made available to the public.


________________
### Instructions

1) Run the "HMM.m" script in MATLAB.

2) Run the "SVM.m" script in MATLAB.(require libsvm)

3) Run the "KNN.m" script in MATLAB.

4) "struct.mat" stores the pre_computed data.

________________
### Notes

If you use the database, please cite this paper

Chunyu Xie, Shangzhen Luan, Hainan Wang, Baochang Zhang: Gesture Recognition Benchmark Based on Mobile Phone. CCBR 2016: 432-440


________________
### Contact

Baochang Zhang

bczhang@buaa.edu.cn

