ע�����
������ csv �ģ�Ҫת���� matlab ��ʽ
�Ķ������ļ��и�����˵��������� label ���͵���������Ҫת���� 01 ���� �� +1/-1�����Ҳ��Ҫת���� 01���� �� +1/-1
�� missing data �������ݲ����ǣ�����
ת������ʹ����д�� label2vector.m ����
����ú��� mat ��ʽ��X Ϊ���룬Y Ϊ�����ÿ��һ�����������Ӽ� miku.mat
Ȼ����� test.m �趨������ͨ��ֻ��� nCenterList����Ϊ����Ԫ��Ŀ������������������ܳ���ѵ��������Ŀ����������
������� test.m ��ע�ͣ��������绰����

�������ش��б�

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

����Դ�������˵����

STSM.m

���㹫ʽ(9)
param: �������� RBFNN ��������ʹ�� trainRBFNN ���
Xtrain: ��������ѵ��������
Q: ���������������� Q ����

�����ӵ� param.STSM ��, �� i �е� j ����ʹ�õ� i �� Q ֵ�������õ��ڵ� j ����������ϵ� sensitivity


trainRBFNN.m

ѵ�� RBFNN
Xtrain, Ytrain: ѵ�����������Ŀ�������������Ԥ��������滯��ÿ��һ��������ÿ�ж�Ӧһ����������/Ŀ�����
param.nCenter: kmeans ������Ŀ�����봦�����ά����������֮��
param.alpha: RBF ��ȿ��Ʋ���

��������:
1. �� nCenter �����ľ������ȵط����������ѵ�����������鿴 centerNum.m
2. ������ʹ�� kmeans �������ĵ�
3. ��������ĵ�����֮ƽ��ֵ������ alpha �õ� RBF �뾶���� v ��û����������Ԫ��ʹ��ͬһ���뾶������
4. ʹ�� exp( - ||x - u||^2 / (2 * v^2)) ��������Ԫ������
5. ʹ����С���˷��� Ytrain ��ΪĿ����ϣ������ W

������������ param.U(����Ԫ���ģ�ÿ��һ��)�� param.V(����Ԫ�뾶����������ÿ�ж�Ӧһ������Ԫ)�� param.W(Ȩ�أ�ÿ�ж�Ӧһ�����)


testRBFNN.m

���� RBFNN ���

preProcesse.m

�����ݸ�ά����ƽ�Ƶ� [-1, +1]


label2vector.m

����ɢ���ǩת���� 0-1 ����
��������࣬ת���� +1/-1


runTest.m

����ʵ��
param.dataFileName: �����ļ���, X ��ʾ���� feature�� Y ��Ŀ�������label �Ѿ�ת���� 01 vector
param.testRatio: �����ȡ�Ĳ��Լ���ռ����
param.nRound: ʵ���ظ�������ÿ�γ�ȡ��ͬ��ѵ��/���Լ���
param.nCenterList: ����Ԫ���б�
param.alphaList: RBF�뾶���������б�
param.Q: Qֵ�б���STSM.m��
