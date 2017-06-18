param = {};

param.dataFileName = 'miku'; % 输入数据文件名，去掉 .m，X 为输入，Y 为输出，必须把标签转为向量
param.nRound = 1; % 重复测试次数
% 所有列表都是列向量
param.nCenterList = (80:80)'; % 隐单元数目列表，不能小于类数目，不能超过训练样本数目 （在 runTest.m 36和37行进行检查）
param.alphaList =  logspace(0,0,1)'; % 半径参数列表 （0.1 ~ 10 等比选取）
param.testRatio = 0.5; % 测试样本抽取比率
param.Q = logspace(-1,-1,1)'; % Q 参数列表 （0.0001 ~ 0.1 等比选取）
result = runTest(param);
resultfile = matfile(sprintf('result-%s-%s.mat', param.dataFileName, date), 'Writable', true);
resultfile.result = result;
resultfile.param = param;
