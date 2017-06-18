param = {};

param.dataFileName = 'miku'; % ���������ļ�����ȥ�� .m��X Ϊ���룬Y Ϊ���������ѱ�ǩתΪ����
param.nRound = 1; % �ظ����Դ���
% �����б���������
param.nCenterList = (80:80)'; % ����Ԫ��Ŀ�б�����С������Ŀ�����ܳ���ѵ��������Ŀ ���� runTest.m 36��37�н��м�飩
param.alphaList =  logspace(0,0,1)'; % �뾶�����б� ��0.1 ~ 10 �ȱ�ѡȡ��
param.testRatio = 0.5; % ����������ȡ����
param.Q = logspace(-1,-1,1)'; % Q �����б� ��0.0001 ~ 0.1 �ȱ�ѡȡ��
result = runTest(param);
resultfile = matfile(sprintf('result-%s-%s.mat', param.dataFileName, date), 'Writable', true);
resultfile.result = result;
resultfile.param = param;
