注意事项：
数据是 csv 的，要转换成 matlab 格式
阅读数据文件夹附带的说明，如果是 label 类型的特征，需要转化成 01 向量 或 +1/-1，输出也是要转化成 01向量 或 +1/-1
有 missing data 的数据暂不考虑，跳过
转化可以使用我写的 label2vector.m 函数
处理好后存成 mat 格式，X 为输入，Y 为输出，每行一个样本，例子见 miku.mat
然后参照 test.m 设定参数，通常只需改 nCenterList，因为隐单元数目不能少于类别数，不能超过训练样本数目，否则会出错
其他详见 test.m 的注释，有问题打电话问我

数据下载处列表

https://archive.ics.uci.edu/ml/datasets/Abalone

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

https://archive.ics.uci.edu/ml/datasets/Credit+Approval

https://archive.ics.uci.edu/ml/datasets/Diabetes

https://archive.ics.uci.edu/ml/datasets/Heart+Disease

https://archive.ics.uci.edu/ml/datasets/Hepatitis

https://archive.ics.uci.edu/ml/datasets/Internet+Advertisements

https://archive.ics.uci.edu/ml/datasets/Ionosphere

https://archive.ics.uci.edu/ml/datasets/ISOLET

https://archive.ics.uci.edu/ml/datasets/Iris

https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

https://archive.ics.uci.edu/ml/datasets/Multiple+Features

https://archive.ics.uci.edu/ml/datasets/Musk+%28Version+2%29

https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification

https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits

https://archive.ics.uci.edu/ml/datasets/Low+Resolution+Spectrometer

https://archive.ics.uci.edu/ml/datasets/Spambase

https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease

https://archive.ics.uci.edu/ml/datasets/Wine

https://archive.ics.uci.edu/ml/datasets/Yeast

https://archive.ics.uci.edu/ml/datasets/Zoo

https://archive.ics.uci.edu/ml/datasets/Corel+Image+Features

https://archive.ics.uci.edu/ml/datasets/CMU+Face+Images

https://archive.ics.uci.edu/ml/datasets/Japanese+Vowels

https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29

https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29

https://archive.ics.uci.edu/ml/datasets/Arcene

https://archive.ics.uci.edu/ml/datasets/Dexter

https://archive.ics.uci.edu/ml/datasets/Wine+Quality

https://archive.ics.uci.edu/ml/datasets/Cardiotocography

https://archive.ics.uci.edu/ml/datasets/Breast+Tissue

https://archive.ics.uci.edu/ml/datasets/Online+Handwritten+Assamese+Characters+Dataset

https://archive.ics.uci.edu/ml/datasets/Amazon+Commerce+reviews+set

https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities

https://archive.ics.uci.edu/ml/datasets/banknote+authentication

https://archive.ics.uci.edu/ml/datasets/Urban+Land+Cover

部分源代码参数说明：

STSM.m

计算公式(9)
param: 参与计算的 RBFNN 的描述，使用 trainRBFNN 获得
Xtrain: 参与计算的训练集输入
Q: 列向量，参与计算的 Q 集合

输出添加到 param.STSM 中, 第 i 行第 j 列是使用第 i 个 Q 值计算所得的在第 j 个网络输出上的 sensitivity


trainRBFNN.m

训练 RBFNN
Xtrain, Ytrain: 训练样本输入和目标输出，必须先预处理和正规化，每行一个样本，每列对应一个特征输入/目标输出
param.nCenter: kmeans 中心数目，必须处于输出维度与样本数之间
param.alpha: RBF 宽度控制参数

计算流程:
1. 将 nCenter 个中心尽量均等地分配给各类别的训练样本，详情看 centerNum.m
2. 在类内使用 kmeans 计算中心点
3. 计算各中心点间距离之平均值，乘以 alpha 得到 RBF 半径参数 v （没错，所有隐单元都使用同一个半径参数）
4. 使用 exp( - ||x - u||^2 / (2 * v^2)) 计算隐单元激活量
5. 使用最小二乘法以 Ytrain 作为目标拟合，计算出 W

计算结果保存在 param.U(隐单元中心，每行一个)， param.V(隐单元半径，列向量，每行对应一个隐单元)， param.W(权重，每列对应一个输出)


testRBFNN.m

计算 RBFNN 输出

preProcesse.m

将数据各维缩放平移到 [-1, +1]


label2vector.m

将离散类标签转化成 0-1 向量
如果是两类，转化成 +1/-1


runTest.m

运行实验
param.dataFileName: 数据文件名, X 表示输入 feature， Y 是目标输出，label 已经转换成 01 vector
param.testRatio: 随机抽取的测试集所占比例
param.nRound: 实验重复次数（每次抽取不同的训练/测试集）
param.nCenterList: 隐单元数列表
param.alphaList: RBF半径缩放因子列表
param.Q: Q值列表（见STSM.m）
