# coding=UTF-8
# ColossalAI main train python.

'''
@File: train.py
@Author: WeiWei
@Time: 2023/2/19
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import colossalai
from colossalai.core import global_context as gpc
import torch
from colossalai.logging import get_dist_logger
import os

'''
python train.py --host <host> --rank <rank> --world_size <world_size> --port <port> --backend <backend>
'''

os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '8888'
os.environ['DATA'] = './datasets/cifar10'

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 后端采用pytorch
colossalai.launch_from_torch(config="config.py")

# 获取多线程logger
logger = get_dist_logger()
logger.set_level(level='info')

# 1. 训练集加载器
train_dataloader = MyTrainDataloader()

# 2. 测试集加载器
test_dataloader = MyTrainDataloader()

# 3. 模型
model = MyModel()

# 4. 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 6. colossalai initial 统一封装
# 返回四个值： engine对象，训练集加载器，测试集加载器，学习率更新器
engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                     optimizer,
                                                                     criterion,
                                                                     train_dataloader,
                                                                     test_dataloader,
                                                                     )
'''
# engine就是对模型、优化、损失函数的封装，它同时继承了模型，优化器和损失函数的一堆方法
engine(inputs)	                                前向计算，等价于model(inputs)
engine.zero_grad()	                            清空梯度，等价于optimizer.zero_grad()
engine.step()	                                更新参数，等价于optimizer.step()
engine.criterion(output, label)	                计算损失，等价于criterion(output, label)
engine.backward(loss)	                        反向传播，等价于loss.backward()
torch.save(engine.model.state_dict(), f=...)	保存模型
engine.model.load_state_dict(torch.load(f=...))	读取模型
engine.train()	                                训练模式，等价于model.train()
engine.eval()	                                评估模式，等价于model.eval()
'''