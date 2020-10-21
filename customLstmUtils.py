import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

BASIC_NAME = "seq_demand_201907_1"  # 输入csv文件名
MODEL_READ = "./data/%s.pth" % BASIC_NAME  # 模型参数读取路径
MODEL_SAVE = "./data/%s.pth" % BASIC_NAME  # 模型参数存储路径

TEST_SIZE = 0.2
OBSERVE_LEN = 12  # 观察期月份数
TRAIN_ROUND = 500  # 训练次数

BATCH_SIZE = 50  # 训练批次样本数
PATIENCE = 3  # n次loss未下降后缩小学习率
INIT_LR = 0.1  # 初始学习率
NUM_LAYERS = 3  # lstm隐藏层层数


def backMonthList(period, end):
    end = end.replace("-", "")
    if len(end) == 6:
        end1 = end + "28"
        while True:
            tmp = (datetime.strptime(end1, '%Y%m%d') + timedelta(days=1)).strftime("%Y%m%d")
            if tmp[0:6] != end:
                end = end1
                break
            else:
                end1 = tmp
    return list(pd.Series(pd.date_range(end=end, periods=period, freq='M')).dt.strftime('%Y%m'))


def monthIndexData(ass):
    return [float(x) for x in ass.split("|")]


def groupByLabel(df):
    data = df.groupby("closing_down").agg({"month_index_0": "count"})[["month_index_0"]].reset_index().values
    df = pd.DataFrame(data=data, columns=["closing_down", "cust_count"])
    print(df)
    count0 = df[df["closing_down"] == 0]["cust_count"].values[0]
    count1 = df[df["closing_down"] == 1]["cust_count"].values[0]
    return count0, count1


def prepareCsvFiles():
    fileName = "./data/%s.csv" % BASIC_NAME
    trainName = "./data/%s_train.csv" % BASIC_NAME
    testName = "./data/%s_test.csv" % BASIC_NAME
    df = pd.read_csv(fileName)
    # 划分训练集和测试集
    train, test = train_test_split(df, test_size=TEST_SIZE)
    print("训练集: %s\n测试集: %s" % (len(train), len(test)))
    train.to_csv(trainName, index=False)
    test.to_csv(testName, index=False)


def readBrandQty(filePath, isTrain):
    columns = ["closing_down"] + ["month_index_%s" % i for i in range(OBSERVE_LEN)]
    df = pd.read_csv(filePath)[columns]
    df["closing_down"] = df["closing_down"].apply(lambda x: int(x))
    data = df.values.tolist()
    brandCount = len(monthIndexData(data[0][2]))
    if isTrain:
        print("训练集：有%s种brand_code，有%s个cust_code。" % (brandCount, len(data)))
    else:
        print("测试集：有%s种brand_code，有%s个cust_code。" % (brandCount, len(data)))
    count0, count1 = groupByLabel(df)
    print("其中%s个0，%s个1。\n" % (count0, count1))

    qtyList, labels = [], []
    for row in data:
        closingDown = row[0]
        monthData = [monthIndexData(r) for r in row[1:]]
        qtyList.append(monthData)
        labels.append(closingDown)
    df = pd.DataFrame({"qty": qtyList, "labels": labels})
    df = df[["qty", "labels"]]
    return df, brandCount, count0, count1


class CustomDataset:
    def __init__(self, filePath, isTrain=True):
        self.brandQty, self.brandCount, self.falseCount, self.trueCount = readBrandQty(filePath, isTrain)

    def __len__(self):
        return len(self.brandQty)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        qty = self.brandQty.iloc[idx, 0]
        qty = np.array(qty).astype('double')
        labels = self.brandQty.iloc[idx, 1]
        labels = np.array(labels)
        sample = {"qty": torch.from_numpy(qty).float(), "labels": torch.from_numpy(labels)}
        return sample


class Lstm(torch.nn.Module):
    def __init__(self, inputSize):
        super(Lstm, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=inputSize, hidden_size=64, num_layers=NUM_LAYERS, batch_first=True, )
        self.out = torch.nn.Linear(64, 2)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out