import mediapipe as mp
import cv2
import numpy as np


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
        dis_4_6 = list_lms[4, :] - list_lms[6, :]
        dis_4_6 = np.sqrt(np.dot(dis_4_6, dis_4_6))

        dis_4_2 = list_lms[4, :] - list_lms[2, :]
        dis_4_2 = np.sqrt(np.dot(dis_4_2, dis_4_2))

        if dis_4_6 < dis_4_2:
            str_guester = "10"
        else:
            str_guester = "Good"

        #str_guester = "Good"

    elif len(up_fingers) == 1 and up_fingers[0] == 20:
        str_guester = "Bad"

    elif len(up_fingers) == 1 and up_fingers[0] == 12:
        str_guester = "FXXX"

    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        str_guester = "2"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 20:  # 修改与6混淆

        dis_4_14 = list_lms[4, :] - list_lms[14, :]
        dis_4_14 = np.sqrt(np.dot(dis_4_14, dis_4_14))

        dis_14_20 = list_lms[14, :] - list_lms[20, :]
        dis_14_20 = np.sqrt(np.dot(dis_14_20, dis_14_20))

        if dis_4_14 < dis_14_20:
            str_guester = "Bad"
        else:
            str_guester = "6"

        # str_guester = "6"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 8:
        dis_4_10 = list_lms[4, :] - list_lms[10, :]
        dis_4_10 = np.sqrt(np.dot(dis_4_10, dis_4_10))

        dis_10_12 = list_lms[10, :] - list_lms[12, :]
        dis_10_12 = np.sqrt(np.dot(dis_10_12, dis_10_12))

        if dis_4_10 < dis_10_12:
            str_guester = "1"
        else:
            str_guester = "8"

        # str_guester = "8"

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
        dis_20_18 = list_lms[20, :] - list_lms[18, :]
        dis_20_18 = np.sqrt(np.dot(dis_20_18, dis_20_18))

        dis_13_17 = list_lms[13, :] - list_lms[17, :]
        dis_13_17 = np.sqrt(np.dot(dis_13_17, dis_13_17))

        if dis_20_18 < dis_13_17:
            str_guester = "8"
        else:
            str_guester = "ROCK"

        # str_guester = "ROCK"

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


def cap_open():
    cap = cv2.VideoCapture(0)
    # 定义手 检测对象
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,  # 静态追踪，低于0.5置信度会再一次跟踪
                          max_num_hands=2,  # 最多有2只手
                          min_detection_confidence=0.5,  # 最小检测置信度
                          min_tracking_confidence=0.5)  # 最小跟踪置信度

    mpDraw = mp.solutions.drawing_utils

    while True:

        # 读取一帧图像
        success, img = cap.read()
        if not success:
            continue
        image_height, image_width, _ = np.shape(img)

        # 转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 得到检测结果
        results = hands.process(imgRGB)
        if results.multi_handedness:
            if len(results.multi_handedness) == 2:  # 检测到两只手
                # for i in range(len(results.multi_handedness)):
                label0 = results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                label1 = results.multi_handedness[1].classification[0].label  # 获得Label判断是哪几手
                index0 = results.multi_handedness[0].classification[0].index  # 获取左右手的索引号
                index1 = results.multi_handedness[1].classification[0].index  # 获取左右手的索引号
                hand0 = results.multi_hand_landmarks[0]  # 根据相应的索引号获取xyz值
                hand1 = results.multi_hand_landmarks[1]  # 根据相应的索引号获取xyz值

                mpDraw.draw_landmarks(img, hand0, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(img, hand1, mpHands.HAND_CONNECTIONS)
                # ---------------hand0----------------------
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
                str_guester1 = get_str_guester(up_fingers, list_lms1)
                # ----------------------------------------------------------------------------------
                cv2.putText(img, ' %s' % (str_guester0), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (str_guester1), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (label0), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (label1), (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)


            ##############################################################################################################################
            elif len(results.multi_handedness) == 1:  # 检测到一只手 :
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

                cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (label), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                for i in ll:
                    pos_x = hand.landmark[i].x * image_width
                    pos_y = hand.landmark[i].y * image_height
                    # 画点
                    cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
            else:
                str_guester = 'no hands'

            cv2.imshow("hands", img)

            # if results.multi_hand_landmarks:
            #
            #     hand1 = results.multi_hand_landmarks[0]
            #
            #     #for hand in results.multi_hand_landmarks:
            #     mpDraw.draw_landmarks(img, hand1, mpHands.HAND_CONNECTIONS)
            #

        key = cv2.waitKey(1) & 0xFF

        # 按键 "q" 退出
        if key == ord('q'):
            break
    cap.release()


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    # 定义手 检测对象
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,  # 静态追踪，低于0.5置信度会再一次跟踪
                          max_num_hands=2,  # 最多有2只手
                          min_detection_confidence=0.5,  # 最小检测置信度
                          min_tracking_confidence=0.5)  # 最小跟踪置信度

    mpDraw = mp.solutions.drawing_utils

    while True:

        # 读取一帧图像
        success, img = cap.read()
        if not success:
            continue
        image_height, image_width, _ = np.shape(img)

        # 转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 得到检测结果
        results = hands.process(imgRGB)
        if results.multi_handedness:
            if len(results.multi_handedness)==2:#   检测到两只手
                #for i in range(len(results.multi_handedness)):
                label0 = results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                label1 = results.multi_handedness[1].classification[0].label  # 获得Label判断是哪几手
                index0 = results.multi_handedness[0].classification[0].index  # 获取左右手的索引号
                index1 = results.multi_handedness[1].classification[0].index  # 获取左右手的索引号
                hand0 = results.multi_hand_landmarks[0]  # 根据相应的索引号获取xyz值
                hand1= results.multi_hand_landmarks[1]  # 根据相应的索引号获取xyz值

                mpDraw.draw_landmarks(img, hand0, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(img, hand1, mpHands.HAND_CONNECTIONS)
#---------------hand0----------------------
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
#-----------------------------------------------------------------------------

#----------------------------hand1---------------------------------------------
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
#----------------------------------------------------------------------------------
                cv2.putText(img, ' %s' % (str_guester0), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (str_guester1), (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (label0), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (label1), (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)


##############################################################################################################################
            elif len(results.multi_handedness)==1:#   检测到一只手 :
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

                cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(img, ' %s' % (label), (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                        cv2.LINE_AA)
                for i in ll:
                    pos_x = hand.landmark[i].x * image_width
                    pos_y = hand.landmark[i].y * image_height
                    # 画点
                    cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)
            else :
                str_guester='no hands'

        cv2.imshow("hands", img)
        a=1
        a=1
                # if results.multi_hand_landmarks:
                #
                #     hand1 = results.multi_hand_landmarks[0]
                #
                #     #for hand in results.multi_hand_landmarks:
                #     mpDraw.draw_landmarks(img, hand1, mpHands.HAND_CONNECTIONS)
                #


        key = cv2.waitKey(1) & 0xFF

        # 按键 "q" 退出
        if key == ord('q'):
            break
    cap.release()












