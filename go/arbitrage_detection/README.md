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

最终所有的数据都会存入mysql数据库中，正负样本默认比例为1:2