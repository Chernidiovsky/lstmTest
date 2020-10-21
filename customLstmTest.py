from customLstmUtils import Lstm, prepareCsvFiles, CustomDataset
from customLstmUtils import TRAIN_ROUND, MODEL_SAVE, MODEL_READ, BASIC_NAME, INIT_LR, BATCH_SIZE, PATIENCE
from torch.utils.data import DataLoader
import torch.optim as optim
import torch


def trainAndTestLstm(split=True, readModel=False, saveModel=True):
    # ------------------------- 加载数据 -------------------------
    if split:  # 重新分配训练集和测试集
        prepareCsvFiles()
    trainSet = CustomDataset("./data/%s_train.csv" % BASIC_NAME, True)
    testSet = CustomDataset("./data/%s_test.csv" % BASIC_NAME, False)
    trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    brandCount = trainSet.brandCount

    # ------------------------- 定义LSTM -------------------------
    lstm = Lstm(brandCount)
    if readModel:
        lstm.load_state_dict(torch.load(MODEL_READ))

    # ------------------------- 损失函数和优化器 -------------------------
    lossFunc = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(lstm.parameters(), lr=INIT_LR, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE, verbose=True)  # 自适应调节lr

    # ------------------------- 训练模型 -------------------------
    for epoch in range(TRAIN_ROUND):
        running_loss = 0.0
        for i, sample in enumerate(trainLoader):
            inputs, labels = sample["qty"], sample["labels"]
            optimizer.zero_grad()
            outputs = lstm(inputs)
            loss = lossFunc(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step(running_loss)
        print("第%s次训练，loss=%.3f" % (epoch + 1, running_loss))
    print("Finished Training")

    # ------------------------- 记录模型 -------------------------
    if saveModel:
        torch.save(lstm.state_dict(), MODEL_SAVE)

    # ------------------------- 测试模型 -------------------------
    testCnt = len(testSet)
    testFalseCnt = testSet.falseCount
    testTrueCnt = testSet.trueCount
    print("测试集总共%s个，%s个0，%s个1。" % (testCnt, testFalseCnt, testTrueCnt))

    right, right1, right0 = 0, 0, 0
    for i, sample in enumerate(testLoader):
        inputs, labels = sample["qty"], sample["labels"]
        outputs = lstm(inputs)
        outputs = torch.max(outputs, 1)[1].data.numpy()
        for j in range(len(outputs)):
            predict, real = int(outputs[j]), int(labels[j])
            if predict == real:
                right += 1
                if predict == 1:
                    right1 += 1
                else:
                    right0 += 1

    print("蒙对的比例: %.2f%%" % (right * 100. / testCnt))
    print("0蒙对的比例: %.2f%%" % (right0 * 100. / testFalseCnt))
    print("1蒙对的比例: %.2f%%" % (right1 * 100. / testTrueCnt))


def merelyTestLstm(testFileName):
    print("./data/%s.csv" % testFileName)
    testSet = CustomDataset("./data/%s.csv" % testFileName, isTrain=False)
    testLoader = DataLoader(testSet, batch_size=4, shuffle=True, num_workers=0)
    brandCount = testSet.brandCount

    lstm = Lstm(brandCount)
    lstm.load_state_dict(torch.load(MODEL_READ))

    testCnt = len(testSet)
    testFalseCnt = testSet.falseCount
    testTrueCnt = testSet.trueCount
    print("测试集总共%s个，%s个0，%s个1。" % (testCnt, testFalseCnt, testTrueCnt))

    right, right1, right0 = 0, 0, 0
    for i, sample in enumerate(testLoader):
        inputs, labels = sample["qty"], sample["labels"]
        outputs = lstm(inputs)
        outputs = torch.max(outputs, 1)[1].data.numpy()
        for j in range(len(outputs)):
            predict, real = int(outputs[j]), int(labels[j])
            if predict == real:
                right += 1
                if predict == 1:
                    right1 += 1
                else:
                    right0 += 1

    print("蒙对的比例: %.2f%%" % (right * 100. / testCnt))
    print("0蒙对的比例: %.2f%%" % (right0 * 100. / testFalseCnt))
    print("1蒙对的比例: %.2f%%" % (right1 * 100. / testTrueCnt))
    print("\n\n\n")


if __name__ == "__main__":
    trainAndTestLstm(split=True, readModel=False, saveModel=True)
    merelyTestLstm("seq_demand_201909_1")
