# coding=UTF-8
# ColossalAI launch file
# ColossalAI training step #2

'''
@File: launch
@Author: WeiWei
@Time: 2023/2/19
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

'''
通过launch可以将配置文件注入系统中，并初始化各种与网络硬件相关的配置。
关于分布式训练有几个比较重要的几个概念：
host: 主训练机的IP
port: 主训练机的端口
host: 训练网络中机器的ID
world size: 网络中机器的数量。
'''

'''
--nproc_per_node : 每个节点GPU的数量
--master_addr : 对应上述 host
--master_port : 对应上述的port
启动训练的命令行与参数如下
$torchrun --nproc_per_node 3 --master_addr localhost --master_port 8001 train.py
'''

import colossalai

# parse arguments
args = colossalai.get_default_parser().parse_args()

# launch distributed environment
colossalai.launch(config='./colossalai_py/config.py',
                  rank=args.rank,
                  world_size=args.world_size,
                  host=args.host,
                  port=args.port,
                  backend=args.backend)
