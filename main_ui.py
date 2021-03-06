# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(950, 650)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 801, 601))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.gridLayout_2.setContentsMargins(-1, -1, 0, -1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gesture_1 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.gesture_1.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gesture_1.sizePolicy().hasHeightForWidth())
        self.gesture_1.setSizePolicy(sizePolicy)
        self.gesture_1.setObjectName("gesture_1")
        self.verticalLayout_3.addWidget(self.gesture_1)
        self.gesture_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gesture_2.sizePolicy().hasHeightForWidth())
        self.gesture_2.setSizePolicy(sizePolicy)
        self.gesture_2.setObjectName("gesture_2")
        self.verticalLayout_3.addWidget(self.gesture_2)
        self.sum = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sum.sizePolicy().hasHeightForWidth())
        self.sum.setSizePolicy(sizePolicy)
        self.sum.setObjectName("sum")
        self.verticalLayout_3.addWidget(self.sum)
        self.gridLayout_2.addLayout(self.verticalLayout_3, 0, 1, 1, 1)
        self.imageShow = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(51)
        sizePolicy.setVerticalStretch(51)
        sizePolicy.setHeightForWidth(self.imageShow.sizePolicy().hasHeightForWidth())
        self.imageShow.setSizePolicy(sizePolicy)
        self.imageShow.setObjectName("imageShow")
        self.gridLayout_2.addWidget(self.imageShow, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(-1, -1, -1, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_filepath = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_filepath.setObjectName("label_filepath")
        self.horizontalLayout.addWidget(self.label_filepath)
        self.text_filepath = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.text_filepath.setObjectName("text_filepath")
        self.horizontalLayout.addWidget(self.text_filepath)
        self.button_filepath = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_filepath.setObjectName("button_filepath")
        self.horizontalLayout.addWidget(self.button_filepath)
        self.button_confirm_filepath = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_confirm_filepath.setObjectName("button_confirm_filepath")
        self.horizontalLayout.addWidget(self.button_confirm_filepath)
        self.button_last_pic = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_last_pic.setObjectName("button_last_pic")
        self.horizontalLayout.addWidget(self.button_last_pic)
        self.button_next_pic = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_next_pic.setObjectName("button_next_pic")
        self.horizontalLayout.addWidget(self.button_next_pic)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_webapi = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_webapi.setObjectName("label_webapi")
        self.horizontalLayout_4.addWidget(self.label_webapi)
        self.text_filepath_toweb = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.text_filepath_toweb.setObjectName("text_filepath_toweb")
        self.horizontalLayout_4.addWidget(self.text_filepath_toweb)
        self.button_select_a_file = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_select_a_file.setObjectName("button_select_a_file")
        self.horizontalLayout_4.addWidget(self.button_select_a_file)
        self.button_upload = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_upload.setObjectName("button_upload")
        self.horizontalLayout_4.addWidget(self.button_upload)
        self.button_switch_cap = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_switch_cap.setObjectName("button_switch_cap")
        self.horizontalLayout_4.addWidget(self.button_switch_cap)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.gridLayout.addLayout(self.verticalLayout_2, 2, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_url = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_url.setObjectName("label_url")
        self.horizontalLayout_2.addWidget(self.label_url)
        self.text_url = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.text_url.setObjectName("text_url")
        self.horizontalLayout_2.addWidget(self.text_url)
        self.button_url_confirm = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.button_url_confirm.setObjectName("button_url_confirm")
        self.horizontalLayout_2.addWidget(self.button_url_confirm)
        self.gridLayout.addLayout(self.horizontalLayout_2, 3, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 950, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.gesture_1.setText(_translate("MainWindow", "??????1???"))
        self.gesture_2.setText(_translate("MainWindow", "??????2???"))
        self.sum.setText(_translate("MainWindow", "????????????"))
        self.imageShow.setText(_translate("MainWindow", "?????????"))
        self.label_filepath.setText(_translate("MainWindow", "?????????????????????"))
        self.button_filepath.setText(_translate("MainWindow", "???????????????"))
        self.button_confirm_filepath.setText(_translate("MainWindow", "??????"))
        self.button_last_pic.setText(_translate("MainWindow", "?????????"))
        self.button_next_pic.setText(_translate("MainWindow", "?????????"))
        self.label_webapi.setText(_translate("MainWindow", "?????????????????????"))
        self.button_select_a_file.setText(_translate("MainWindow", "????????????"))
        self.button_upload.setText(_translate("MainWindow", "??????"))
        self.button_switch_cap.setText(_translate("MainWindow", "???????????????"))
        self.label_url.setText(_translate("MainWindow", "webapi_??????url???"))
        self.button_url_confirm.setText(_translate("MainWindow", "??????"))
