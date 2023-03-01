# coding=UTF-8
# ColossalAI training config file
# ColossalAI training step #1

'''
@File: config
@Author: WeiWei
@Time: 2023/2/19
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

'''
# colossalai 参数含义解释
parallel	并行配置，是一个字典，可配置的子项为数据并行、流水线并行和序列并行	https://www.colossalai.org/zh-Hans/docs/basics/configure_parallelization
fp16	混合精度策略	https://www.colossalai.org/zh-Hans/docs/features/mixed_precision_training
gradient_accumulation	梯度累计次数	https://www.colossalai.org/zh-Hans/docs/features/gradient_accumulation
clip_grad_norm	梯度裁剪范数	https://www.colossalai.org/zh-Hans/docs/features/gradient_clipping
gradient_handler	自定义处理梯度同步的类	https://www.colossalai.org/zh-Hans/docs/features/gradient_handler
MOE_MODEL_PARALLEL_SIZE	一个进程中的混合专家模型数量	https://www.colossalai.org/zh-Hans/
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

from colossalai.amp import AMP_TYPE

BATCH_SIZE = 128     # 批次大小
NUM_EPOCHS = 100    # 训练10轮

fp16 = dict(
  mode=AMP_TYPE.TORCH     # AMP后端是pytorh
)

parallel = dict(          # 并行策略，请注意，pipline的取值和tensor的size的乘积为你GPU的数量（此例中为2 * 4 = 8）
    pipeline=1,
    tensor=dict(size=1, mode='2d')
)