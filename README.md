1 安装

去官网https://pytorch.org/寻找对应系统和版本的安装指令。

验证安装成功，检查三个必备import：

from torch.utils.data import DataLoader
import torch.optim as optim
import torch


2 定义LSTM

LSTM定义语句：

    class Lstm(torch.nn.Module):

        def __init__(self, inputSize):
            super(Lstm, self).__init__()
            self.rnn = torch.nn.LSTM(input_size=inputSize, hidden_size=64, num_layers=3, batch_first=True, )
            
            # input_size为输入层大小，表示烟品数，是一个样本在一个时间点上的输入，也就是矩阵的一行的长度
            # hidden_size为隐藏层大小，取64。
            # num_layers为隐藏层层数，取3。这两个是自由设置的参数，但会影响计算性能和结果。
            # 输入数据的长度从一个输入层到三个隐藏层再到一个输出层的变化为： 烟品数 => 64 =>64 => 64 => 2
            
            self.out = torch.nn.Linear(64, 2)

        def forward(self, x):
            r_out, (h_n, h_c) = self.rnn(x, None)
            # LSTM一个时序的输入和输出，None表示无初始态。
            out = self.out(r_out[:, -1, :])
            # -1表示取得最后一个时序的输出。
            return out


3 训练样本
3.1 数据集DataLoader

	trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

trainSet是样本集，每个样本包含一个二维Tensor（矩阵：12 x 烟品数）和一个标签（整数0或1）；
batch_size是单次训练批次的数量，表示把样本集按batch_size的大小分割，分批次训练的目的是把大数据集切割成小批量来提高训练效率（批训练介绍）；
shuffle表示打乱顺序；
num_workers必须=0否则报错。

3.2 损失函数和优化器

    lossFunc = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(lstm.parameters(), lr=0.1, momentum=0.9)
SGD是最基本的梯度下降函数，介绍参见https://morvanzhou.github.io/tutorials/machine-learning/torch/3-06-A-speed-up-learning/。

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        
学习率lr类似梯度下降的补偿，越大收敛越快，但可能在极小值周围震荡。因此设置动态学习率，原理是连续patience个epoch的训练结果的损失函数都没有减小后，按比例缩小学习率。

3.3 训练过程

    for epoch in range(TRAIN_ROUND):
        running_loss = 0.0
        for sample in trainLoader:
        # 第一个for表示重复训练，第二个for是处理DataLoader分好的批次的每个批次。
            inputs, labels = sample["qty"], sample["labels"]
            # 取得矩阵和标签
            optimizer.zero_grad()
            # 本批次的梯度清零
            outputs = lstm(inputs)
            # 计算lstm输出
            loss = lossFunc(outputs, labels)
            loss.backward()
            optimizer.step()
            # 根据损失函数反向更新lstm的参数
            running_loss += loss.item()
            # 累加各批次损失函数值
        scheduler.step(running_loss)
        # lr调整
    print("第%s次训练，loss=%.3f" % (epoch + 1, running_loss))


4 结果测试

    torch.save(lstm.state_dict(), MODEL_SAVE)

训练完毕后保存模型为pth文件。

    for i, sample in enumerate(testLoader):
        inputs, labels = sample["qty"], sample["labels"]
        outputs = lstm(inputs)
        outputs = torch.max(outputs, 1)[1].data.numpy()

对比labels和outputs，即可获得准确率。
