# 图书推荐系统

## 项目简介

这是一个图书推荐系统，使用的数据集为高校图书馆近几年的借阅数据，通过使用DSSM深度语义匹配模型对其中的用户进行个性化推荐，本仓库是项目的后端实现，具体的前端界面可访问[图书管理系统](https://mikus.love/book/login)，本项目旨在通过个性化推荐系统实现高校图书馆资源的充分利用。

项目的github地址：[BookRecSystem](https://github.com/Mikeaser/BookRecSystem)

## 运行说明

**教程位于项目目录下的文件BookRecTutorial.ipynb当中**

环境要求位于项目目录下的目录Requirements中

包含windows与linux系统的环境信息

想要复刻环境，终端运行

```shell
pip install -r requirements.txt
```

**注意**

annoy包需要系统带有C++编译环境，终端执行

```bash
sudo apt update
sudo apt install cmake gcc g++ make python3-dev
```
