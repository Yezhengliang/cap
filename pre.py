#!usr/bin/env python  
# -*- coding:utf-8 _*-
# @author:Torres Ye
# @file: pre.py 
# @version:
# @time: 2019/05/25 
# @email:yzlview@163.com
import cv2, os
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, train_test_split

map_list = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'j': 9, 'k': 10, 'm': 11, 'n': 12, 'p': 13,
            'q': 14, 'r': 15, 's': 16, 't': 17, 'u': 18, 'v': 19, 'w': 20, 'x': 21, 'y': 22, 'z': 23}
new_dict = {v: k for k, v in map_list.items()}


class pred():
    def __init__(self, imgpath):
        self.imgpath = imgpath
        self.result = ''

    def crop(self, thresh1):
        file_context = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

        height, width = opening.shape[:2]
        v = [0] * width
        a = 0

        # 垂直投影：统计并存储每一列的黑点数
        for x in range(0, width):
            for y in range(0, height):
                if thresh1[y, x] == 0:
                    a = a + 1
                else:
                    continue
            v[x] = a
            a = 0

        result = []
        # print(v)
        flag = True
        index_start_list = []
        index_end_list = []
        for index, value in enumerate(v):
            if value and flag:  # 起始
                flag = False
                index_start_list.append(index)
                continue
            elif not flag and not value:  # 结束
                flag = True
                index_end_list.append(index)
                continue
            else:
                pass
        if len(index_end_list) == 4:
            for start, end in zip(index_start_list, index_end_list):
                # print(start, end)
                if end - start < 5 or end - start > 18:
                    pass
                else:
                    img_temp = thresh1[0:20, start:end]
                    size = 18 - (end - start)
                    img_temp = cv2.copyMakeBorder(img_temp, 0, 0, 0, size, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    img_np = np.array(img_temp).astype(np.int32).reshape((1, 360))
                    img_np[img_np < 255] = 1
                    img_np[img_np == 255] = 0
                    result.append(img_np)

        else:
            for w in range(4):
                img_temp = thresh1[0:20, w * 18:(w + 1) * 18]
                img_np = np.array(img_temp).astype(np.int32).reshape((1, 360))
                img_np[img_np < 255] = 1
                img_np[img_np == 255] = 0
                result.append(img_np)
        if result:
            return result

    def ceratedataset(self, ):
        dataset = np.ones((1, 361))
        for root, sub_dirs, files in os.walk('./ocr'):
            for dirs in sub_dirs:
                for fileName in os.listdir('./ocr' + '/' + dirs):
                    img = cv2.imread('./ocr' + '/' + dirs + '/' + fileName, 0)
                    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    # for m in self.crop(thresh1):
                    img_np = np.array(thresh1).astype(np.int32).reshape((1, 360))
                    img_np[img_np < 255] = 1
                    img_np[img_np == 255] = 0
                    dataset = np.append(dataset, np.append(map_list[dirs], img_np).reshape((1, 361)), axis=0)
        np.savetxt('dataset_me.txt', dataset)

    def train(self):
        data = np.genfromtxt('dataset_me.txt', delimiter=' ')
        x = data[1:, 1:]  # 数据特征
        y = data[1:, 0].astype(int)  # 标签
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
        svc = SVC(kernel='rbf', class_weight='balanced', )
        c_range = np.logspace(-5, 15, 11, base=2)
        gamma_range = np.logspace(-9, 3, 13, base=2)
        # 网格搜索交叉验证的参数范围，cv=3,3折交叉
        param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
        grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
        # 训练模型
        clf = grid.fit(x_train, y_train)
        # 计算测试集精度
        score = grid.score(x_test, y_test)
        print('精度为%s' % score)
        joblib.dump(clf, './letter_me.pkl')

    def pred(self, img):
        clf = joblib.load('./letter_me.pkl')
        oneLetter = clf.predict(img)[0]
        return new_dict[oneLetter]

    def main(self):
        img = cv2.imread(self.imgpath, 0)
        ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img_file = self.crop(thresh1)
        result = []
        for m in img_file:
            # print(self.pred(m))
            result.append(self.pred(m))
        result = ''.join(result)
        print(result)
        return result


if __name__ == '__main__':
    main = pred('./img/test.jpg')
    main.main()
