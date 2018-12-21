# DataCastle-2018中国（合肥）城市大数据与人工智能创新应用大赛
## 【化工产品品质预测】 [比赛页面戳这里](http://www.hfdatacity.com/common/cmpt/%E5%8C%96%E5%B7%A5%E4%BA%A7%E5%93%81%E5%93%81%E8%B4%A8%E6%99%BA%E8%83%BD%E9%A2%84%E6%B5%8B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html "化工产品品质预测")

### 比赛代码由两个队员单独折腾再将结果进行了融合，有点混乱，能跑就好~

- 第一分支：
【main1.ipynb】
   1. 运行Load Data模块读入原始数据；

   2. 运行Preprocessing 1模块给检测数据加上对应的批次id，产生train_with_id和test_with_id文件；

   3. 运行Preprocessing 2模块提取检测数据特征，提交的模型仅使用了每一批次的均值，因为在A榜上测试效果好；

   4. 运行Train 1模块准备数据，移除特征值全为空的样本，以均值填补缺失值；

   5. 运行Train 2-2*: “lightGBM 全数据集训练模型，直接产生测试结果”；

   6. 运行Train 2-4*: “SVR 全数据集训练模型，直接产生测试结果”；

   7. 运行Train 2-5*: “KNN距离加权 全数据集训练模型，直接产生测试结果”

   8. simple stacking “模块输出测试结果”，由于比赛要求预测产品的5个指标，上述每个基模型针对每个指标都训练了一个模型，考虑到5个指标之间可能存在的相互影响关系，以3*5=15个输出为特征，进行了第二轮训练再次拟合训练集指标，我们称之为simple stacking，得到A榜0.28992的结果；


- 第二分支:
   
   【preprocess.py】
   
   9. 运行预处理preprocess.py 得到含样本对应批次时间均值特征的训练数据train_mean.csv和test_mean.csv；

   【train_round1.py】

   10. 运行train_round1.py 得到lightgbm模型以及svr模型对应的训练集预测结果和测试集预测结果train_data_lgb.csv,test_data_lgb.csv,train_data_svr.csv,test_data_svr.csv；

   【train_round2.py】
   
   11. 运行train_round2.py 得到mlp模型对第一轮训练两个模型预测的simple stacking融合结果训练得到A榜0.288429的结果；


- 一、二分支融合：
   
   【main1.ipynb】
   
   12. 运行result merge模块，以A榜得分倒数为权重加权融合，得到最终结果A榜0.280405。
   
- 思路  
>【特征提取】  
直接使用官方提供的螺旋门压缩后的不等长时间序列作为原始数据，统计每个批次各检测点均值、方差、最大值、最小值、中值等数据作为训练特征；  
>
>【第一轮，单模型】：  
lightGBM，基于学习算法的决策树梯度boosting框架，效果好，速度快；
>
>SVR，支持向量回归；
>
>KNN-weightedRegression，以各样本的特征向量的余弦距离倒数为权重，对各样本质量指标加权求和得出测试样本的预测结果；
>
>【第二轮，5指标再预测】
考虑到预测出的5个质量指标可能存在的线性或非线性关系，利用第一轮的模型在训练集上跑出的结果Y_pred作为新特征X'，训练集label作为Y',再训练一个回归模型M拟合Y'，然后测试集第一轮的结果过一遍模型M，得到更准确的预测结果；
>
>【第三轮，最终融合】
我们训练了两个模型分支：
第一分支是lightGBM+SVR+KNN(第一轮)/lightGBM(第二轮)，A榜评分0.28992；
第二分支是lightGBM+SVR(第一轮)/MLP(第二轮)，A榜评分0.288429；
>
>以两个分支A榜评分的倒数为权重加权求和，得到最终融合预测结果，A榜评分0.280405，B榜评分0.36951，排名第6。