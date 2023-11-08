"""
==========================================
@绘制训练过程
@训练过程可视化
==========================================
"""
import datetime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import re
import os
import numpy as np
#忽略append报错
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


matplotlib.rc("font", family='SimHei')  # 设置中文字体

#设置时间
date = datetime.date.today().strftime('%Y-%m-%d')


# 日志文件路径
json_path = "/home/niu/mmsegmentation/work_dirs/WHUDataset-hrnet/20231026_154830/vis_data/20231026_154830.json"
#从日志文件中提取网络名称
net_name = json_path.split('WHUDataset-')[1].split('/')[0]



with open(json_path, "r") as f:
    json_list = f.readlines()
# print(eval(json_list[4]))

df_train = pd.DataFrame()
df_test = pd.DataFrame()
for each in json_list[:-1]:
    if 'aAcc' in each:
        df_test = df_test.append(eval(each), ignore_index=True)
    else:
        df_train = df_train.append(eval(each), ignore_index=True)

dir_name = './chart_output/'+date+net_name
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    
df_train.to_csv('./chart_output/'+date+net_name+'/训练日志-训练集.csv', index=False)
df_test.to_csv('./chart_output/'+date+net_name+'/训练日志-测试集.csv', index=False)

# 定义所有线形、颜色、标记
random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick',
          'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen',
          'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray',
          'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue',
          'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid',
          'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred',
          'deeppink', 'hotpink']
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D",
           "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
linestyle = ['--', '-.', '-']


def get_line_arg():
    """
    随机产生一种绘图线型
    :return: 线形
    """
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg


def show_train_loss():
    metrics = ['loss', 'decode.loss_ce', 'aux.loss_ce']
    plt.figure(figsize=(16, 8))

    x = df_train['step']
    for y in metrics:
        try:
            plt.plot(x, df_train[y], label=y, **get_line_arg())
        except:
            pass
    #纵坐标0-1
    plt.ylim(0,1)
    plt.tick_params(labelsize=20)
    plt.xlabel('step', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('训练集损失函数', fontsize=25)
    plt.legend(fontsize=20)
    plt.savefig('./chart_output/'+date+net_name+'/训练集损失函数.pdf', dpi=120, bbox_inches='tight')
    plt.show()

def show_train_acc():
    metrics = ['decode.acc_seg', 'aux.acc_seg']
    plt.figure(figsize=(16, 8))

    x = df_train['step']
    for y in metrics:
        try:
            plt.plot(x, df_train[y], label=y, **get_line_arg())
        except:
            pass
    plt.ylim(0,100)
    plt.tick_params(labelsize=20)
    plt.xlabel('step', fontsize=20)
    plt.ylabel('Metrics', fontsize=20)
    plt.title('训练集准确率', fontsize=25)
    plt.legend(fontsize=20)
    plt.savefig('./chart_output/'+date+net_name+'/训练集准确率.pdf', dpi=120, bbox_inches='tight')
    plt.show()


def show_test():
    metrics = ['aAcc', 'mIoU', 'mAcc', 'mDice', 'mFscore', 'mPrecision', 'mRecall']
    plt.figure(figsize=(16, 8))

    x = df_test['step']
    for y in metrics:
        try:
            plt.plot(x, df_test[y], label=y, **get_line_arg())
        except:
            pass

    plt.tick_params(labelsize=20)
    plt.ylim([0, 100])
    plt.xlabel('step', fontsize=20)
    plt.ylabel('Metrics', fontsize=20)
    plt.title('测试集评估指标', fontsize=25)
    plt.legend(fontsize=20)
    plt.savefig('./chart_output/'+date+net_name+'/测试集分类评估指标.pdf', dpi=120, bbox_inches='tight')
    plt.show()


# 类别名
class_list = ['background', 'buildings']
# 日志文件路径
log_path = json_path
print(log_path)

with open(log_path, 'r') as f:
    logs = f.read()


def transform_table_line(raw):
    """
    定义正则表达式
    :param raw:
    :return:
    """
    raw = list(map(lambda x: x.split('|'), raw))
    raw = list(map(
      lambda row: list(map(
          lambda col: float(col.strip()),
          row
      )),
      raw
    ))
    return raw


# 横轴：起点，总训练迭代次数，单位间隔
x = range(0, 60000, 500)
metrics_json = {}
for each_class in class_list:  # 遍历每个类别
    re_pattern = r'\s+{}.*?\|(.*)?\|'.format(each_class)  # 定义该类别的正则表达式
    metrics_json[each_class] = {}
    metrics_json[each_class]['re_pattern'] = re.compile(re_pattern)

# 匹配
for each_class in class_list: # 遍历每个类别
    find_string = re.findall(metrics_json[each_class]['re_pattern'], logs)  # 粗匹配
    find_string = transform_table_line(find_string)  # 精匹配
    metrics_json[each_class]['metrics'] = find_string


def show_each_class():
    for each_class in class_list:  # 遍历每个类别
        each_class_metrics = np.array(metrics_json[each_class]['metrics'])

        plt.figure(figsize=(16, 8))

        for idx, each_metric in enumerate(['IoU', 'Acc', 'Dice', 'Fscore', 'Precision', 'Recall']):

            try:
                plt.plot(x, each_class_metrics[:, idx], label=each_metric, **get_line_arg())
            except:
                pass

        plt.tick_params(labelsize=20)
        plt.ylim([0, 100])
        plt.xlabel('step', fontsize=20)
        plt.ylabel('Metrics', fontsize=20)
        plt.title('类别{}在测试集上的评估指标'.format(each_class), fontsize=25)

        plt.legend(fontsize=20)

        # plt.savefig('类别 {} 训练过程评估指标.pdf'.format(each_class), dpi=120, bbox_inches='tight')

        plt.show()


if __name__ == '__main__':
    show_train_loss()
    show_train_acc()
    show_test()
    #show_each_class()

