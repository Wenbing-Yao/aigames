# 代码说明

0. snake_v0/1/2 是很久以前写的贪吃蛇游戏
1. snake_env.py 中为改编后，封装的 snake 的运行环境
2. wfs.py 为使用广度优先搜索算法运行; 其中play函数会返回运行时的状态记录 x 为状态， y 为 方向，其中保存运行结果的代码被注释掉了
3. simple_mlp.py 使用wfs.py 运行得到的数据集进行训练
4. mlp-play.py  使用 simple_mlp.py 训练得到的结果运行游戏
5. ga-origin.py 为使用遗传算法进行训练，每次迭代一次都会玩一次游戏
   1. 这里直接使用了遗传算法库 pygad
   2. 会保存每一代的网络
6. gaplay.py 代码基本和ga-origin.py 一样，我复制了用来加载指定generation的保存的结果
   1. 运行方法   gaplay.py [number of generation], 比如 python gaplay.py 40 表示运行第40代结果
7. hc.py 为当前实验的最优方法