import pandas as pd
import os


# 平衡0、1，抽样以保证不超内存
file_dir = "E:\\Download\\"
file_list = []
for root, dirs, files in os.walk(file_dir):
    for filesPath in files:
        file_list.append(os.path.join(root, filesPath))

df = pd.DataFrame()
for f in file_list:
    print(f)
    recognizePic = pd.read_csv(f)
    df = df.append(recognizePic)
    df = df.reset_index(drop=True)
    del recognizePic


df0 = df[df["closing_down"] == 0]
df1 = df[df["closing_down"] == 1]
print(len(df0))
print(len(df1))

num = 50000
df0 = df0.sample(n=num)
df = df0.append(df1)
df = df.reset_index(drop=True)
df.to_csv("D:\\PycharmProjects\\testRepoPy3\\lstmTest\\data\\201912gd.csv")

df = pd.read_csv("D:\\PycharmProjects\\testRepoPy3\\lstmTest\\data\\201912all.csv")
df0 = df[df["closing_down"] == 0]
df1 = df[df["closing_down"] == 1]
print(len(df0))
print(len(df1))
num = 40000
df = df.sample(n=num)
df0 = df[df["closing_down"] == 0]
df1 = df[df["closing_down"] == 1]
print(len(df0))
print(len(df1))
df.to_csv("D:\\PycharmProjects\\testRepoPy3\\lstmTest\\data\\201912all_1.csv")
