# 搜集Arbi数据

go的版本为1.23.4

1、安装Mysql数据库8.0版本

2、创建自己的数据库

3、导入`sql`包下的`arbitrary.sql`文件，会创建好各个表

4、修改`data_collection`包下个文件的配置：

- bsc浏览器的token
- bsc archive节点的连接地址
- 数据库的连接地址
- 其他...

5、运行 `go run main.go`

最终所有的数据都会存入mysql数据库中，正负样本默认比例为1:2，其中80%会用作训练集，另外的，针对剩余的20%会对负样本进行扩充，最终测试集中正负样本约为1:13。


## 说明

arbi_collector.go：套利数据分析与收集

constants.go：常量

db_connection.go：数据库连接管理

dto.go：dto模型管理

eth_client.go：bsc区块链连接管理

model.go：数据库模型

model_helper.go：模型转换帮助函数，model->dto

runner.go：执行器，里面包含完整数据收集以及非套利交易收集的具体过程
