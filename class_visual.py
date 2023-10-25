import csv
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 读取日志文件
# 并按类别写入csv文件
def extract_class_index(log_path):
    with open(log_path, 'r') as f:
        log_lines = f.readlines()

        # 定义正则表达式模式
        pattern = r'\|\s+(?P<class>\w+)\s+\|\s+(?P<iou>\d+\.\d+)\s+\|\s+(?P<acc>\d+\.\d+)\s+\|\s+(?P<dice>\d+\.\d+)\s+\|\s+(?P<fscore>\d+\.\d+)\s+\|\s+(?P<precision>\d+\.\d+)\s+\|\s+(?P<recall>\d+\.\d+)\s+\|'

        # 创建CSV文件并写入表头
        #net_name = log_path.split('WHUDataset-')[1].split('/')[0]
        #print(net_name)
        net_name='FCN'

        folder_path = './chart_output/'

        # 遍历文件夹
        for root, dirs, files in os.walk(folder_path):


            dateofthisnet=os.path.basename(log_path)
            dateofthisnet=dateofthisnet[:8]
            chart_path = folder_path  +dateofthisnet + "_"+ net_name

            os.makedirs(chart_path,exist_ok=True)

            csv_path = chart_path + '/metrics.csv'

            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['Class', 'IoU', 'Acc', 'Dice', 'Fscore', 'Precision', 'Recall']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for line in log_lines:
                    # 匹配正则表达式
                    match = re.match(pattern, line)
                    if match:
                        # 提取指标
                        class_name = match.group('class')
                        iou = float(match.group('iou'))
                        acc = float(match.group('acc'))
                        dice = float(match.group('dice'))
                        fscore = float(match.group('fscore'))
                        precision = float(match.group('precision'))
                        recall = float(match.group('recall'))

                        if class_name == 'buildings':

                            writer.writerow({'Class': class_name, 'IoU': iou, 'Acc': acc, 'Dice': dice, 'Fscore': fscore,
                                             'Precision': precision, 'Recall': recall})
                        else:
                            writer.writerow({'Class': class_name, 'IoU': iou, 'Acc': acc, 'Dice': dice, 'Fscore': fscore,
                                             'Precision': precision, 'Recall': recall})
            return csv_path

def plot_class(csv_path, class_name):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 提取class名为"buildings"的行
    buildings_df = df[df['Class'] == class_name]

    # 添加iters列
    iters = range(500, 60001, 500)
    buildings_df = buildings_df.assign(iters=iters)

    # 绘制折线图
    plt.plot(buildings_df['iters'], buildings_df['IoU'], label='iou')
    plt.plot(buildings_df['iters'], buildings_df['Acc'], label='Acc')
    plt.plot(buildings_df['iters'], buildings_df['Fscore'], label='Fscore')
    plt.plot(buildings_df['iters'], buildings_df['Precision'], label='Precision')

    # 设置图例、标题和轴标签
    plt.legend()
    plt.title('index for Class ' + class_name)
    plt.xlabel('iters')
    plt.ylabel('Value')
    
    
    #plt.show()
    # 保存为PDF文件
    pdf_path = csv_path.rsplit('/', 1)[0] + '/' + class_name + '_plot.pdf'
    plt.savefig(pdf_path)
    
    # 关闭图形
    plt.close()

    print("图形已保存为PDF文件。")


if __name__ == '__main__':
    csv_path1 = extract_class_index("/home/niu/mmsegmentation/work_dirs/WHUDataset-FCN/20231020_110542/20231020_110542.log")
    
    csv_path = csv_path1
    class_name = 'buildings'
    plot_class(csv_path, class_name)
