# from flask import Flask
# from flask import request
# # from flask_script import Manager
# from gevent import pywsgi
# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     return '<h1>Home</h1>'

# @app.route('/signin', methods=['GET'])
# def signin_form():
#     return '''<form action="/signin" method="post">
#               <p><input name="username"></p>
#               <p><input name="password" type="password"></p>
#               <p><button type="submit">Sign In</button></p>
#               </form>'''

# @app.route('/signin', methods=['POST'])
# def signin():
#     # 需要从request对象读取表单内容：
#     if request.form['username']=='admin' and request.form['password']=='password':
#         return '<h3>Hello, admin!</h3>'
#     return '<h3>Bad username or password.</h3>'

# if __name__ == '__main__':
#     server = pywsgi.WSGIServer(('', 5000), app)
#     server.serve_forever()
#     app.run()
#     # manager=Manager(app)
#     # manager.run()   #非开发者模式
from flask import Flask
from flask import request
import mediapipe as mp
import cv2
import numpy as np
import requests
<<<<<<< HEAD
def get_angle(v1,v2):
    angle = np.dot(v1,v2)/(np.sqrt(np.sum(v1*v1))*np.sqrt(np.sum(v2*v2)))
    angle = np.arccos(angle)/3.14*180
    
    return angle 
    
    
def get_str_guester(up_fingers,list_lms):
    
    if len(up_fingers)==1 and up_fingers[0]==8:
        
        v1 = list_lms[6]-list_lms[7]
        v2 = list_lms[8]-list_lms[7]
        
        angle = get_angle(v1,v2)
    
        if angle<160:
            str_guester = "9"
        else:
            str_guester = "1"
    
    elif len(up_fingers)==1 and up_fingers[0]==4:
        str_guester = "Good"
    
    elif len(up_fingers)==1 and up_fingers[0]==20:
        str_guester = "Bad"
        
    elif len(up_fingers)==1 and up_fingers[0]==12:
        str_guester = "FXXX"

    elif len(up_fingers)==2 and up_fingers[0]==8 and up_fingers[1]==12:
        str_guester = "2"
        
    elif len(up_fingers)==2 and up_fingers[0]==4 and up_fingers[1]==20:
        str_guester = "6"
        
    elif len(up_fingers)==2 and up_fingers[0]==4 and up_fingers[1]==8:
        str_guester = "8"
    
    elif len(up_fingers)==3 and up_fingers[0]==8 and up_fingers[1]==12 and up_fingers[2]==16:
        str_guester = "3"
    
    elif len(up_fingers)==3 and up_fingers[0]==4 and up_fingers[1]==8 and up_fingers[2]==12:

        dis_8_12 = list_lms[8,:] - list_lms[12,:]
        dis_8_12 = np.sqrt(np.dot(dis_8_12,dis_8_12))
        
        dis_4_12 = list_lms[4,:] - list_lms[12,:]
        dis_4_12 = np.sqrt(np.dot(dis_4_12,dis_4_12))
        
        if dis_4_12/(dis_8_12+1) <3:
            str_guester = "7"
        
        elif dis_4_12/(dis_8_12+1) >5:
            str_guester = "Gun"
        else:
            str_guester = "7"
            
    elif len(up_fingers)==3 and up_fingers[0]==4 and up_fingers[1]==8 and up_fingers[2]==20:
        str_guester = "ROCK"
    
    elif len(up_fingers)==4 and up_fingers[0]==8 and up_fingers[1]==12 and up_fingers[2]==16 and up_fingers[3]==20:
        str_guester = "4"
    
    elif len(up_fingers)==5:
        str_guester = "5"
        
    elif len(up_fingers)==0:
        str_guester = "10"
    
    else:
        str_guester = " "
        
    return str_guester


# def judge_api(imgsrc):
#     # response = requests.get(imgsrc)
#     # img = Image.open(BytesIO(response.content))
#     file = requests.get(imgsrc)
#     img = cv2.imdecode(np.fromstring(file.content, np.uint8), 1)
#     # img = cv2.imread(imgsrc)
#     # cap = cv2.VideoCapture(0)
#     # 定义手 检测对象
#     mpHands = mp.solutions.hands
#     hands = mpHands.Hands(model_complexity=1,
#                             min_detection_confidence=0.5,min_tracking_confidence=0.5)
#     mpDraw = mp.solutions.drawing_utils

#     while True:

#         # 读取一帧图像
#         # success, img = cap.read()
#         # if not success:
#         #     continue
#         image_height, image_width, _ = np.shape(img)
        
#         # 转换为RGB
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # 得到检测结果
#         results = hands.process(imgRGB)
        
#         if results.multi_hand_landmarks:
#             hand = results.multi_hand_landmarks[0]
            
#             mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)
            
#             # 采集所有关键点的坐标
#             list_lms = []    
#             for i in range(21):
#                 pos_x = hand.landmark[i].x*image_width
#                 pos_y = hand.landmark[i].y*image_height
#                 list_lms.append([int(pos_x),int(pos_y)])
            
#             # 构造凸包点
#             list_lms = np.array(list_lms,dtype=np.int32)
#             hull_index = [0,1,2,3,6,10,14,19,18,17,10]
#             hull = cv2.convexHull(list_lms[hull_index,:])
#             # 绘制凸包
#             cv2.polylines(img,[hull], True, (0, 255, 0), 2)
                
#             # 查找外部的点数
#             n_fig = -1
#             ll = [4,8,12,16,20] 
#             up_fingers = []
            
#             for i in ll:
#                 pt = (int(list_lms[i][0]),int(list_lms[i][1]))
#                 dist= cv2.pointPolygonTest(hull,pt,True)
#                 if dist <0:
#                     up_fingers.append(i)
            
#             # print(up_fingers)
#             # print(list_lms)
#             # print(np.shape(list_lms))
#             str_guester = get_str_guester(up_fingers,list_lms)
#             # 输出到终端
#             print(str_guester)
#             # cv2.putText(img,' %s'%(str_guester),(90,90),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,0),4,cv2.LINE_AA)
            
                
#         #     for i in ll:
#         #         pos_x = hand.landmark[i].x*image_width
#         #         pos_y = hand.landmark[i].y*image_height
#         #         # 画点
#         #         cv2.circle(img, (int(pos_x),int(pos_y)), 3, (0,255,255),-1)
#         # cv2.namedWindow("hands", cv2.WINDOW_NORMAL)
#         # cv2.imshow("hands",img)
#         # cv2.resizeWindow("hands", 500, 500)
#         # key =  cv2.waitKey(1) & 0xFF   
#         return str_guester
#         break
def judge_api(imgsrc):
    file = requests.get(imgsrc)
    img = cv2.imdecode(np.fromstring(file.content, np.uint8), 1)
    # img = cv2.imread(imgsrc)
    # cap = cv2.VideoCapture(0)
    # 定义手 检测对象
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(model_complexity=1,
                            min_detection_confidence=0.5,min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    while True:

        # 读取一帧图像
        # success, img = cap.read()
        # if not success:
        #     continue
=======
import base64
import json
import os


def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    return angle


def get_str_guester(up_fingers, list_lms):
    if len(up_fingers) == 1 and up_fingers[0] == 8:

        v1 = list_lms[6] - list_lms[7]
        v2 = list_lms[8] - list_lms[7]

        angle = get_angle(v1, v2)

        if angle < 160:
            str_guester = "9"
        else:
            str_guester = "1"

    elif len(up_fingers) == 1 and up_fingers[0] == 4:
        str_guester = "Good"

    elif len(up_fingers) == 1 and up_fingers[0] == 20:
        str_guester = "Bad"

    elif len(up_fingers) == 1 and up_fingers[0] == 12:
        str_guester = "FXXX"

    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        str_guester = "2"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 20:
        str_guester = "6"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 8:
        str_guester = "8"

    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16:
        str_guester = "3"

    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 12:

        dis_8_12 = list_lms[8, :] - list_lms[12, :]
        dis_8_12 = np.sqrt(np.dot(dis_8_12, dis_8_12))

        dis_4_12 = list_lms[4, :] - list_lms[12, :]
        dis_4_12 = np.sqrt(np.dot(dis_4_12, dis_4_12))

        if dis_4_12 / (dis_8_12 + 1) < 3:
            str_guester = "7"

        elif dis_4_12 / (dis_8_12 + 1) > 5:
            str_guester = "Gun"
        else:
            str_guester = "7"

    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 20:
        str_guester = "ROCK"

    elif len(up_fingers) == 4 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16 and up_fingers[
        3] == 20:
        str_guester = "4"

    elif len(up_fingers) == 5:
        str_guester = "5"

    elif len(up_fingers) == 0:
        str_guester = "10"

    else:
        str_guester = " "

    return str_guester


def getByte(img):
    path = "/www/wwwroot/py_test/img.jpg"
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str


# 1为web端网址，0为本地图片上传
def judge_api(imageFile, flag):
    img = 1
    if (flag == 1):
        file = requests.get(imageFile)
        img = cv2.imdecode(np.fromstring(file.content, np.uint8), 1)
    else:
        img = imageFile
    # 定义手 检测对象
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(model_complexity=1,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    while True:
>>>>>>> jcy
        image_height, image_width, _ = np.shape(img)

        # 转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 得到检测结果
        results = hands.process(imgRGB)
        if results.multi_handedness:
<<<<<<< HEAD
            if len(results.multi_handedness)==2:#   检测到两只手
                #for i in range(len(results.multi_handedness)):
=======
            if len(results.multi_handedness) == 2:  # 检测到两只手
                # for i in range(len(results.multi_handedness)):
>>>>>>> jcy
                label0 = results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                label1 = results.multi_handedness[1].classification[0].label  # 获得Label判断是哪几手
                index0 = results.multi_handedness[0].classification[0].index  # 获取左右手的索引号
                index1 = results.multi_handedness[1].classification[0].index  # 获取左右手的索引号
                hand0 = results.multi_hand_landmarks[0]  # 根据相应的索引号获取xyz值
<<<<<<< HEAD
                hand1= results.multi_hand_landmarks[1]  # 根据相应的索引号获取xyz值

                mpDraw.draw_landmarks(img, hand0, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(img, hand1, mpHands.HAND_CONNECTIONS)
#---------------hand0----------------------
=======
                hand1 = results.multi_hand_landmarks[1]  # 根据相应的索引号获取xyz值

                mpDraw.draw_landmarks(img, hand0, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(img, hand1, mpHands.HAND_CONNECTIONS)
                # ---------------hand0----------------------
>>>>>>> jcy
                # 采集所有关键点的坐标
                list_lms0 = []
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
                str_guester0 = get_str_guester(up_fingers, list_lms0)
<<<<<<< HEAD
#-----------------------------------------------------------------------------

#----------------------------hand1---------------------------------------------
=======
                # -----------------------------------------------------------------------------

                # ----------------------------hand1---------------------------------------------
>>>>>>> jcy
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
                str_guester1 = get_str_guester(up_fingers, list_lms1)
<<<<<<< HEAD
#----------------------------------------------------------------------------------
                cv2.putText(img, ' %s' % (str_guester0), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (str_guester1), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (label0), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (label1), (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                return "Left Hand:"+str_guester0 +",Right Hand:"+str_guester1+"."

##############################################################################################################################
            elif len(results.multi_handedness)==1:#   检测到一只手 :
=======

                # ----------------------------------------------------------------------------------
                cv2.putText(img, ' webserverreturn', (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                # cv2.putText(img, ' %s' % (str_guester0), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                #             cv2.LINE_AA)
                # cv2.putText(img, ' %s' % (str_guester1), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                #             cv2.LINE_AA)
                # cv2.putText(img, ' %s' % (label0), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                #             cv2.LINE_AA)
                # cv2.putText(img, ' %s' % (label1), (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                #             cv2.LINE_AA)
                cv2.imwrite("/www/wwwroot/py_test/img.jpg", img)
                img_str = getByte(img)
                return '{"str_guester0":' + '"' + str_guester0 + '",' + '"str_guester1":' + '"' + str_guester1 + '",' + '"label0":' + '"' + label0 + '",' + 'label1":' + '"' + label1 + '",' + '"img_str":' + '"' + img_str + '"' + "}"

            ##############################################################################################################################
            elif len(results.multi_handedness) == 1:  # 检测到一只手 :
>>>>>>> jcy
                label = results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                index = results.multi_handedness[0].classification[0].index  # 获取左右手的索引号
                hand = results.multi_hand_landmarks[0]

                # for hand in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

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
                str_guester = get_str_guester(up_fingers, list_lms)
<<<<<<< HEAD

                cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (label), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                        cv2.LINE_AA)
=======
                cv2.putText(img, ' webserverreturn', (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                # cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                #             cv2.LINE_AA)
                # cv2.putText(img, ' %s' % (label), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                #         cv2.LINE_AA)
>>>>>>> jcy
                for i in ll:
                    pos_x = hand.landmark[i].x * image_width
                    pos_y = hand.landmark[i].y * image_height
                    # 画点
                    cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
<<<<<<< HEAD
                return "The Hand:"+str_guester+"."
            else :
                str_guester='no hands' 
                return "There is "+str_guester+"."
            # cv2.imshow("hands", img)
            break
app = Flask(__name__)

@app.route('/')
def index():
  pic_src=request.args.get('src')
  str = judge_api(pic_src)
  return str
#   return 'test hello world'

if __name__ == '__main__':
  app.run()
=======
                cv2.imwrite("/www/wwwroot/py_test/img.jpg", img)
                img_str = getByte(img)

                return '{"str_guester":' + '"' + str_guester + '",' + '"label":' + '"' + label + '",' + '"img_str":' + '"' + img_str + '"' + "}"
            else:
                str_guester = 'no hands'
                cv2.putText(img, ' webserverreturn', (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)

                cv2.imwrite("/www/wwwroot/py_test/img.jpg", img)
                img_str = getByte(img)
                return '{"img_str":' + '"' + img_str + '"' + "}"
                return "There is " + str_guester + "."
            # cv2.imshow("hands", img)
        else:
            str_guester = 'no hands'
            cv2.putText(img, ' webserverreturn', (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                        cv2.LINE_AA)
            cv2.imwrite("/www/wwwroot/py_test/img.jpg", img)
            img_str = getByte(img)
            return '{"img_str":' + '"' + img_str + '"' + "}"
            return "There is " + str_guester + "."
        break


app = Flask(__name__)


@app.route('/webapi', methods=['GET', 'POST'])
def web_api():
    pic_src = request.args.get('src')
    str = judge_api(pic_src, 1)
    return str


@app.route('/upload', methods=['GET', 'POST'])
def index():
    json_list = json.loads(request.data)
    img_str = json_list['img_str']
    img_decode_ = img_str.encode('ascii')  # ascii编码
    img_decode = base64.b64decode(img_decode_)  # base64解码
    img_np = np.frombuffer(img_decode, np.uint8)  # 从byte数据读取为np.array形式
    img = cv2.imdecode(img_np, cv2.COLOR_RGB2BGR)  # 转为OpenCV形式
    str = judge_api(img, 0)
    return str


#   return 'test hello world'

if __name__ == '__main__':
    app.run()
>>>>>>> jcy
