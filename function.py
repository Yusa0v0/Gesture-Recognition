import json

import requests
from PyQt5.QtCore import QTimer
import base64


from main_ui import Ui_MainWindow
import handtracking
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import mediapipe as mp
import numpy as np
import filetype
import os
import openpyxl
import time
from tkinter import *
from tkinter.filedialog import askdirectory
import threading

global imageListIndex
global imageList
def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str
class UI(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(UI, self).__init__(parent)
        self.timer_image=QTimer()
        self.timer_cap = QTimer()
        # self.cap =0
        self.imageListIndex=0
        self.imageList=[]
        self.setupUi(self)
        self.action_init()
        print("process start")
    def action_init(self):
        self.button_filepath.clicked.connect(lambda : self.select_filepath())
        self.button_confirm_filepath.clicked.connect(lambda :self.confirm_filepath())
        self.button_select_a_file.clicked.connect(lambda : self.select_file())
        self.button_upload.clicked.connect(lambda : self.upload_image())
        self.button_switch_cap.clicked.connect(lambda : self.switch_timer_cap())
        self.button_last_pic.setEnabled(False)
        self.button_next_pic.setEnabled(False)
        self.button_upload.setEnabled(False)
        self.button_confirm_filepath.setEnabled(False)
        self.button_last_pic.clicked.connect(lambda :self.last_pic())
        self.button_next_pic.clicked.connect(lambda :self.next_pic())
        self.button_url_confirm.clicked.connect(lambda :self.image_url())

        self.imageShow.setScaledContents(True)

        # self.timer_image.timeout.connect(lambda :self.timer_image_fun())
        self.timer_cap.timeout.connect(lambda :self.cap_model())

    # 选择文件
    def select_filepath(self):
        filePath=QtWidgets.QFileDialog.getExistingDirectory(self)
        self.text_filepath.setText(filePath)
        if (len(self.text_filepath.text()) != 0):
            self.button_confirm_filepath.setEnabled(True)
    def select_file(self):
        file,filetype = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "选取文件", "./","jpg Files (*.jpg)")
        self.text_filepath_toweb.setText(str(file))
        if(len(self.text_filepath_toweb.text())!=0):
            self.button_upload.setEnabled(True)


    # 处理图片并显示
    def show_image(self,image_name):
        kind = filetype.guess(image_name)
        if kind is None or kind.extension != 'jpg':
            print("invalid type for " + image_name)
            return

        image = cv2.imread(image_name)

        # adjust size
        # height, width = image.shape[:2]
        # height_coefficient = height // 720
        # width_coefficient = width // 1280
        # zoom_coefficient = 0
        # if height_coefficient > zoom_coefficient:
        #     zoom_coefficient = height_coefficient
        # if width_coefficient > zoom_coefficient:
        #     zoom_coefficient = width_coefficient
        # if zoom_coefficient > 0:
        #     image = cv2.resize(image, (int(width / zoom_coefficient), int(height / zoom_coefficient)),
        #                        interpolation=cv2.INTER_CUBIC)

        img=image
        while True:
            image_height, image_width, _ = np.shape(img)

            # 转换为RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 得到检测结果
            results = self.hands.process(imgRGB)
            if results.multi_handedness:
                if len(results.multi_handedness) == 2:  # 检测到两只手
                    # for i in range(len(results.multi_handedness)):
                    label0 = results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                    label1 = results.multi_handedness[1].classification[0].label  # 获得Label判断是哪几手
                    index0 = results.multi_handedness[0].classification[0].index  # 获取左右手的索引号
                    index1 = results.multi_handedness[1].classification[0].index  # 获取左右手的索引号
                    hand0 = results.multi_hand_landmarks[0]  # 根据相应的索引号获取xyz值
                    hand1 = results.multi_hand_landmarks[1]  # 根据相应的索引号获取xyz值

                    self.mpDraw.draw_landmarks(img, hand0, self.mpHands.HAND_CONNECTIONS)
                    self.mpDraw.draw_landmarks(img, hand1, self.mpHands.HAND_CONNECTIONS)
                    # ---------------hand0----------------------
                    # 采集所有关键点的坐标
                    list_lms0 = []
                    ll = [4, 8, 12, 16, 20]
                    for i in range(21):
                        pos_x = hand0.landmark[i].x * image_width
                        pos_y = hand0.landmark[i].y * image_height
                        list_lms0.append([int(pos_x), int(pos_y)])
                    for i in ll:
                        pos_x = hand0.landmark[i].x * image_width
                        pos_y = hand0.landmark[i].y * image_height
                        # 画点
                        cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
                    # 构造凸包点
                    list_lms0 = np.array(list_lms0, dtype=np.int32)
                    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                    hull = cv2.convexHull(list_lms0[hull_index, :])
                    # 绘制凸包
                    cv2.polylines(img, [hull], True, (0, 255, 0), 2)

                    # 查找外部的点数
                    n_fig = -1
                    ll = [4, 8, 12, 16, 20]
                    up_fingers = []

                    for i in ll:
                        pt = (int(list_lms0[i][0]), int(list_lms0[i][1]))
                        dist = cv2.pointPolygonTest(hull, pt, True)
                        if dist < 0:
                            up_fingers.append(i)

                    # print(up_fingers)
                    # print(list_lms)
                    # print(np.shape(list_lms))
                    str_guester0 = handtracking.get_str_guester(up_fingers, list_lms0)
                    # -----------------------------------------------------------------------------

                    # ----------------------------hand1---------------------------------------------
                    # 采集所有关键点的坐标
                    list_lms1 = []
                    for i in range(21):
                        pos_x = hand1.landmark[i].x * image_width
                        pos_y = hand1.landmark[i].y * image_height
                        list_lms1.append([int(pos_x), int(pos_y)])
                    for i in ll:
                        pos_x = hand1.landmark[i].x * image_width
                        pos_y = hand1.landmark[i].y * image_height
                        # 画点
                        cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
                    # 构造凸包点
                    list_lms1 = np.array(list_lms1, dtype=np.int32)
                    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                    hull = cv2.convexHull(list_lms1[hull_index, :])
                    # 绘制凸包
                    cv2.polylines(img, [hull], True, (0, 255, 0), 2)

                    # 查找外部的点数
                    n_fig = -1
                    ll = [4, 8, 12, 16, 20]
                    up_fingers = []

                    for i in ll:
                        pt = (int(list_lms1[i][0]), int(list_lms1[i][1]))
                        dist = cv2.pointPolygonTest(hull, pt, True)
                        if dist < 0:
                            up_fingers.append(i)

                    # print(up_fingers)
                    # print(list_lms)
                    # print(np.shape(list_lms))
                    str_guester1 = handtracking.get_str_guester(up_fingers, list_lms1)
                    # ----------------------------------------------------------------------------------
                    # cv2.putText(img, ' %s' % (str_guester0), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                    #             cv2.LINE_AA)
                    # cv2.putText(img, ' %s' % (str_guester1), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                    #             cv2.LINE_AA)
                    # cv2.putText(img, ' %s' % (label0), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                    #             cv2.LINE_AA)
                    # cv2.putText(img, ' %s' % (label1), (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                    #             cv2.LINE_AA)

                    self.gesture_1.setText("手势1：" + label1 + "," + str_guester0)
                    self.gesture_2.setText("手势2：" + label0 + "," + str_guester1)
                    if (len(str_guester0) == 1 and len(
                            str_guester1) == 1 and str_guester0 >= '0' and str_guester0 <= '9' and str_guester1 >= '0' and str_guester1 <= '9'):
                        sum = int(str_guester0) + int(str_guester1)
                        self.sum.setText("数字和：" + str(sum))
                    else:
                        self.sum.setText("数字和：")

                ##############################################################################################################################
                elif len(results.multi_handedness) == 1:  # 检测到一只手 :
                    label = results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                    index = results.multi_handedness[0].classification[0].index  # 获取左右手的索引号
                    hand = results.multi_hand_landmarks[0]

                    # for hand in results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)

                    # 采集所有关键点的坐标
                    list_lms = []
                    for i in range(21):
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        list_lms.append([int(pos_x), int(pos_y)])
                    # 构造凸包点
                    list_lms = np.array(list_lms, dtype=np.int32)
                    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                    hull = cv2.convexHull(list_lms[hull_index, :])
                    # 绘制凸包
                    cv2.polylines(img, [hull], True, (0, 255, 0), 2)

                    # 查找外部的点数
                    n_fig = -1
                    ll = [4, 8, 12, 16, 20]
                    up_fingers = []

                    for i in ll:
                        pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                        dist = cv2.pointPolygonTest(hull, pt, True)
                        if dist < 0:
                            up_fingers.append(i)

                    # print(up_fingers)
                    # print(list_lms)
                    # print(np.shape(list_lms))
                    str_guester = handtracking.get_str_guester(up_fingers, list_lms)

                    # cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                    #             cv2.LINE_AA)
                    # cv2.putText(img, ' %s' % (label), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                    #             cv2.LINE_AA)
                    for i in ll:
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        # 画点
                        cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
                    if (label == "Right"):
                        if(str_guester!=" "):
                            self.gesture_1.setText("手势1：" + "Left" + ", " + str_guester)
                        else:
                            self.gesture_1.setText("手势1：" + "Left" )
                    else:
                        if (str_guester !=" "):
                            self.gesture_1.setText("手势1：" + "Right" + ", " + str_guester)
                        else:
                            self.gesture_1.setText("手势1：" + "Right")
                    self.gesture_2.setText("手势2：")
                    self.sum.setText("数字和：")
                else:
                    str_guester = 'no hands'

            else:
                print("no find")
                self.gesture_1.setText("手势1：")
                self.gesture_2.setText("手势2：")
                self.sum.setText("数字和：")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            showimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3,QtGui.QImage.Format_RGB888)
            self.imageShow.setPixmap(QtGui.QPixmap.fromImage(showimg))
            break

    def confirm_filepath(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,  # 静态追踪，低于0.5置信度会再一次跟踪
                                        max_num_hands=2,  # 最多有2只手
                                        min_detection_confidence=0.5,  # 最小检测置信度
                                        min_tracking_confidence=0.5)  # 最小跟踪置信度
        self.mpDraw = mp.solutions.drawing_utils
        # ---------------------------------------------------------------------------#
        # 功能
        fpath = self.text_filepath.text()
        print(fpath)

        myFiles = os.listdir(fpath)
        self.imageList.clear()
        for eachOne in myFiles:
            size = os.path.getsize(os.path.join(fpath, eachOne))
            if size <= 1:
                print ("{0} is {1} bytes.".format(eachOne, size))
            else:
                self.imageList.append(eachOne)
        # self.imageList = os.listdir(fpath)
        if('test_list.xlsx' in self.imageList):
            self.imageList.remove('test_list.xlsx')
        print(self.imageList)
        print(len(self.imageList))

        self.imageListIndex = 0
        self.button_last_pic.setEnabled(False)
        if(len(self.imageList)>1):
            self.button_next_pic.setEnabled(True)
        print(fpath + '/' + self.imageList[self.imageListIndex])
        self.show_image(fpath + '/' + self.imageList[self.imageListIndex])
        # ---------------------------------------------------------------------------#
        # auto test function
        # fpath = self.text_filepath.text()
        # fpath_cpy=fpath
        # sonPathList = os.listdir(fpath)
        #
        # bk = openpyxl.Workbook()
        # i=1
        # for pathIndex in sonPathList:
        #     bk.create_sheet(pathIndex)
        #     bk._active_sheet_index=i
        #     i=i+1
        #     sh1 = bk.active
        #     sh1.append(['编号', '文件名', '手势1', "手势2", "数字和"])
        #     no=1
        #     fpath=fpath_cpy+ '/'+pathIndex
        #     self.imageListIndex=0
        #     self.imageList.clear()
        #     myFiles = os.listdir(fpath)
        #     for eachOne in myFiles:
        #         size = os.path.getsize(os.path.join(fpath, eachOne))
        #         if size <= 1:
        #             print("{0} is {1} bytes.".format(eachOne, size))
        #         else:
        #             self.imageList.append(eachOne)
        #     while(self.imageListIndex<len(self.imageList)):
        #         print(fpath + '/' + self.imageList[self.imageListIndex])
        #         self.show_image(fpath + '/' + self.imageList[self.imageListIndex])
        #         info =[no,self.imageList[self.imageListIndex],self.gesture_1.text()[4:],self.gesture_2.text()[4:],self.sum.text()[4:]]
        #         if(len(self.gesture_2.text())>6):
        #             # os.remove(fpath+ '/' + self.imageList[self.imageListIndex])
        #             a=1
        #         elif(len(self.gesture_1.text())<10):
        #             # os.remove(fpath+ '/' + self.imageList[self.imageListIndex])
        #             a=1
        #         elif(pathIndex=="013-bad" and self.gesture_1.text().find("6")!=-1):
        #
        #             os.remove(fpath + '/' + self.imageList[self.imageListIndex])
        #         else:
        #             sh1.append(info)
        #             no+=1
        #         self.imageListIndex+=1
        #         # time.sleep(0.1)
        # bk.save(fpath_cpy+'/test_list.xlsx')
        # print("result has been saved.")
        # self.button_next_pic.setEnabled(False)

    def last_pic(self):
        fpath = self.text_filepath.text()
        if(self.imageListIndex!=0):
            self.imageListIndex-=1
            self.show_image(fpath + '/' + self.imageList[self.imageListIndex])
            self.button_next_pic.setEnabled(True)

        if(self.imageListIndex==0):
            self.button_last_pic.setEnabled(False)
    def next_pic(self):
        fpath = self.text_filepath.text()
        if (self.imageListIndex != len(self.imageList)-1 ):
            self.imageListIndex += 1
            self.button_last_pic.setEnabled(True)
            print(fpath + '/' + self.imageList[self.imageListIndex])
            self.show_image(fpath + '/' + self.imageList[self.imageListIndex])
        if (self.imageListIndex == len(self.imageList)-1 ):
            self.button_next_pic.setEnabled(False)

    # web端
    def upload_image(self):
        path = self.text_filepath_toweb.text()
        url = "http://49.234.156.195/upload"
        img_str = getByte(path)
        # print(img_str)
        files = {
            "img_str":  img_str
        }
        json_mod = json.dumps(files)
        response = requests.post(url=url, data=json_mod)
        jsonList = json.loads(str(response.text))
        print(len(jsonList))
        if(len(jsonList)>=4):
            str_guester0=jsonList['str_guester0']
            str_guester1=jsonList['str_guester1']
            label0=jsonList['label0']
            label1=jsonList['label1']
            self.gesture_1.setText("手势1：" + label1 + "," + str_guester0)
            self.gesture_2.setText("手势2：" + label0 + "," + str_guester1)
            if (len(str_guester0) == 1 and len(
                    str_guester1) == 1 and str_guester0 >= '0' and str_guester0 <= '9' and str_guester1 >= '0' and str_guester1 <= '9'):
                sum = int(str_guester0) + int(str_guester1)
                self.sum.setText("数字和：" + str(sum))
            else:
                self.sum.setText("数字和：")
        elif(len(jsonList)>=2):
            str_guester = jsonList['str_guester']
            label = jsonList['label']
            if (label == "Right"):
                self.gesture_1.setText("手势1：" + "Left" + "," + str_guester)
            else:
                self.gesture_1.setText("手势1：" + "Right" + "," + str_guester)
            self.gesture_2.setText("手势2：")
            self.sum.setText("数字和：")

        img_str = jsonList['img_str']
        print(img_str)
        img_decode_ = img_str.encode('ascii')  # ascii编码
        img_decode = base64.b64decode(img_decode_)  # base64解码
        # img_decode = base64.b64decode(img_str)  # base64解码

        img_np = np.frombuffer(img_decode, np.uint8)  # 从byte数据读取为np.array形式
        img = cv2.imdecode(img_np, cv2.COLOR_RGB2BGR)  # 转为OpenCV形式
        # cv2.imshow("hands", img)
        # 显示图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.imageShow.setPixmap(QtGui.QPixmap.fromImage(showimg))
        # print(response.text)
        print(response.status_code)

    def image_url(self):
        url = "http://49.234.156.195/webapi"
        src = self.text_url.text()
        url=url+"?src="+str(src)
        print(url)
        response = requests.get(url=url)
        # self.gesture_1.setText(response.text)
        jsonList = json.loads(str(response.text))
        print(len(jsonList))
        if (len(jsonList) >= 4):
            str_guester0 = jsonList['str_guester0']
            str_guester1 = jsonList['str_guester1']
            label0 = jsonList['label0']
            label1 = jsonList['label1']
            self.gesture_1.setText("手势1：" + label1 + "," + str_guester0)
            self.gesture_2.setText("手势2：" + label0 + "," + str_guester1)
            if (len(str_guester0) == 1 and len(
                    str_guester1) == 1 and str_guester0 >= '0' and str_guester0 <= '9' and str_guester1 >= '0' and str_guester1 <= '9'):
                sum = int(str_guester0) + int(str_guester1)
                self.sum.setText("数字和：" + str(sum))
            else:
                self.sum.setText("数字和：")
        elif (len(jsonList) >= 2):
            str_guester = jsonList['str_guester']
            label = jsonList['label']
            if (label == "Right"):
                self.gesture_1.setText("手势1：" + "Left" + "," + str_guester)
            else:
                self.gesture_1.setText("手势1：" + "Right" + "," + str_guester)
            self.gesture_2.setText("手势2：")
            self.sum.setText("数字和：")

        img_str = jsonList['img_str']
        print(img_str)
        img_decode_ = img_str.encode('ascii')  # ascii编码
        img_decode = base64.b64decode(img_decode_)  # base64解码
        # img_decode = base64.b64decode(img_str)  # base64解码

        img_np = np.frombuffer(img_decode, np.uint8)  # 从byte数据读取为np.array形式
        img = cv2.imdecode(img_np, cv2.COLOR_RGB2BGR)  # 转为OpenCV形式
        # cv2.imshow("hands", img)
        # 显示图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.imageShow.setPixmap(QtGui.QPixmap.fromImage(showimg))
    # finished
    def switch_timer_cap(self):
        if not self.timer_cap.isActive():
            self.button_last_pic.setEnabled(False)
            self.button_next_pic.setEnabled(False)
            self.button_switch_cap.setText('关闭摄像头')
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(5, 30)  # 帧率
            self.cap.set(3, 1280)  # 帧宽
            self.cap.set(4, 720)  # 帧高
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(static_image_mode=False,  # 静态追踪，低于0.5置信度会再一次跟踪
                                  max_num_hands=2,  # 最多有2只手
                                  min_detection_confidence=0.5,  # 最小检测置信度
                                  min_tracking_confidence=0.5)  # 最小跟踪置信度
            self.mpDraw = mp.solutions.drawing_utils
            self.timer_cap.start(30)
        else:
            self.button_last_pic.setEnabled(False)
            self.button_next_pic.setEnabled(False)
            self.cap.release()
            self.button_switch_cap.setText('启用摄像头')
            self.imageShow.clear()
            self.imageShow.setText("图片：")
            self.timer_cap.stop()

        # if(self.button_switch_cap.text=="启用摄像头"):
        #     self.button_switch_cap.setText('关闭摄像头')
        #     self.timer_cap.start(30)
        # else:
        #     self.timer_cap.stop()
        #     self.button_switch_cap.setText('启用摄像头')
    def cap_model(self):

        # 定义手 检测对象
        # 读取一帧图像
        success, img = self.cap.read()
        print(success)
        if not success:
            print("ERROR")
            self.timer_cap.stop()
            return
        image_height, image_width, _ = np.shape(img)
        # print("step1")
        # 转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 得到检测结果
        results = self.hands.process(imgRGB)

        if results.multi_handedness:
            if len(results.multi_handedness) == 2:  # 检测到两只手
                print("two hands")
                # for i in range(len(results.multi_handedness)):
                label0 = results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                label1 = results.multi_handedness[1].classification[0].label  # 获得Label判断是哪几手
                index0 = results.multi_handedness[0].classification[0].index  # 获取左右手的索引号
                index1 = results.multi_handedness[1].classification[0].index  # 获取左右手的索引号
                hand0 = results.multi_hand_landmarks[0]  # 根据相应的索引号获取xyz值
                hand1 = results.multi_hand_landmarks[1]  # 根据相应的索引号获取xyz值

                self.mpDraw.draw_landmarks(img, hand0, self.mpHands.HAND_CONNECTIONS)
                self.mpDraw.draw_landmarks(img, hand1, self.mpHands.HAND_CONNECTIONS)
                # ---------------hand0----------------------
                # 采集所有关键点的坐标
                list_lms0 = []
                ll = [4, 8, 12, 16, 20]
                for i in range(21):
                    pos_x = hand0.landmark[i].x * image_width
                    pos_y = hand0.landmark[i].y * image_height
                    list_lms0.append([int(pos_x), int(pos_y)])
                for i in ll:
                    pos_x = hand0.landmark[i].x * image_width
                    pos_y = hand0.landmark[i].y * image_height
                    # 画点
                    cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
                # 构造凸包点
                list_lms0 = np.array(list_lms0, dtype=np.int32)
                hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                hull = cv2.convexHull(list_lms0[hull_index, :])
                # 绘制凸包
                cv2.polylines(img, [hull], True, (0, 255, 0), 2)

                # 查找外部的点数
                n_fig = -1
                ll = [4, 8, 12, 16, 20]
                up_fingers = []

                for i in ll:
                    pt = (int(list_lms0[i][0]), int(list_lms0[i][1]))
                    dist = cv2.pointPolygonTest(hull, pt, True)
                    if dist < 0:
                        up_fingers.append(i)

                # print(up_fingers)
                # print(list_lms)
                # print(np.shape(list_lms))
                str_guester0 = handtracking.get_str_guester(up_fingers, list_lms0)
                # -----------------------------------------------------------------------------

                # ----------------------------hand1---------------------------------------------
                # 采集所有关键点的坐标
                list_lms1 = []
                for i in range(21):
                    pos_x = hand1.landmark[i].x * image_width
                    pos_y = hand1.landmark[i].y * image_height
                    list_lms1.append([int(pos_x), int(pos_y)])
                for i in ll:
                    pos_x = hand1.landmark[i].x * image_width
                    pos_y = hand1.landmark[i].y * image_height
                    # 画点
                    cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
                # 构造凸包点
                list_lms1 = np.array(list_lms1, dtype=np.int32)
                hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                hull = cv2.convexHull(list_lms1[hull_index, :])
                # 绘制凸包
                cv2.polylines(img, [hull], True, (0, 255, 0), 2)

                # 查找外部的点数
                n_fig = -1
                ll = [4, 8, 12, 16, 20]
                up_fingers = []

                for i in ll:
                    pt = (int(list_lms1[i][0]), int(list_lms1[i][1]))
                    dist = cv2.pointPolygonTest(hull, pt, True)
                    if dist < 0:
                        up_fingers.append(i)

                # print(up_fingers)
                # print(list_lms)
                # print(np.shape(list_lms))
                str_guester1 = handtracking.get_str_guester(up_fingers, list_lms1)
                # ----------------------------------------------------------------------------------
                # cv2.putText(img, ' %s' % (str_guester0), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0),
                #             4,
                #             cv2.LINE_AA)
                # cv2.putText(img, ' %s' % (str_guester1), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0),
                #             4,
                #             cv2.LINE_AA)
                # cv2.putText(img, ' %s' % (label0), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                #             cv2.LINE_AA)
                # cv2.putText(img, ' %s' % (label1), (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                #             cv2.LINE_AA)
                self.gesture_1.setText("手势1：" + label1 + "," + str_guester0)
                self.gesture_2.setText("手势2：" + label0 + "," + str_guester1)
                if (len(str_guester0) == 1 and len(str_guester1) == 1 and str_guester0>='0' and str_guester0<='9'  and str_guester1>='0' and str_guester1<='9'):
                    sum = int(str_guester0) + int(str_guester1)
                    self.sum.setText("数字和："+str(sum))
                else:
                    self.sum.setText("数字和：")


            ##############################################################################################################################
            elif len(results.multi_handedness) == 1:  # 检测到一只手 :
                print("one hand")

                label = results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                index = results.multi_handedness[0].classification[0].index  # 获取左右手的索引号
                hand = results.multi_hand_landmarks[0]

                # for hand in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)

                # 采集所有关键点的坐标
                list_lms = []
                for i in range(21):
                    pos_x = hand.landmark[i].x * image_width
                    pos_y = hand.landmark[i].y * image_height
                    list_lms.append([int(pos_x), int(pos_y)])
                # 构造凸包点
                list_lms = np.array(list_lms, dtype=np.int32)
                hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                hull = cv2.convexHull(list_lms[hull_index, :])
                # 绘制凸包
                cv2.polylines(img, [hull], True, (0, 255, 0), 2)

                # 查找外部的点数
                n_fig = -1
                ll = [4, 8, 12, 16, 20]
                up_fingers = []

                for i in ll:
                    pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                    dist = cv2.pointPolygonTest(hull, pt, True)
                    if dist < 0:
                        up_fingers.append(i)

                # print(up_fingers)
                # print(list_lms)
                # print(np.shape(list_lms))
                str_guester = handtracking.get_str_guester(up_fingers, list_lms)

                # cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                #             cv2.LINE_AA)
                # cv2.putText(img, ' %s' % (label), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                #             cv2.LINE_AA)
                if (label == "Right"):
                    self.gesture_1.setText("手势1：" + "Left" + "," + str_guester)
                else:
                    self.gesture_1.setText("手势1：" + "Right" + "," + str_guester)
                self.gesture_2.setText("手势2：")
                self.sum.setText("数字和：")
                for i in ll:
                    pos_x = hand.landmark[i].x * image_width
                    pos_y = hand.landmark[i].y * image_height
                    # 画点
                    cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
            else:
                str_guester = 'no hands'
                print("no hands")
                self.gesture_1.setText("手势1：")
                self.gesture_2.setText("手势2：")
                self.sum.setText("数字和：")
            # img = cv2.imdecode(np.fromfile(directory[0], dtype=np.uint8), -1)  # 路径含有中文 需要用imdecode
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            showimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            self.imageShow.setPixmap(QtGui.QPixmap.fromImage(showimg))
        else:
            print("no find")
            self.gesture_1.setText("手势1：")
            self.gesture_2.setText("手势2：")
            self.sum.setText("数字和：")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            showimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            self.imageShow.setPixmap(QtGui.QPixmap.fromImage(showimg))

        key = cv2.waitKey(1) & 0xFF
        # self.cap.release()
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = UI()
    MainWindow.setWindowTitle("手势识别")
    MainWindow.show()
    sys.exit(app.exec_())  # 关闭窗口
