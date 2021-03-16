import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from dialog_1 import Ui_Dialog
import os
import glob
import os
import sys
import shutil
from PyQt5.QtCore import QThread, pyqtSignal
import yaml

class WorkThread(QThread):

    _signal = pyqtSignal()
    def __init__(self,command):
        super(WorkThread,self).__init__()
        self.command = command
    def run(self):
        os.system(self.command)
        self._signal.emit()
    

MODEL = ["SSL_t","SSL_s",
        "mmdet_air_before","mmdet_ship_before","mmdet_air_after","mmdet_ship_after",
        "fewshot_aug","fewshot_dota",
        "active_before","active_after",
        "attr"]
IMAGE_FORMAT = ["bmp", "jpg", "jpeg"]

class TMP(QWidget, Ui_Dialog):
    def __init__(self):
        QWidget.__init__(self)
        Ui_Dialog.__init__(self)
        self.setupUi(self)
        self.setupConnect(self)
        self.page = 0
        self.gallery_size = 0
        self.output_path = None
        self.config_path = './config.yaml'
        self.load_config()
    
    def load_config(self):
        with open(self.config_path) as f:            
            self.config = yaml.load(f)
        print(self.config)

    def openfolder_model(self):
        model_dir = QFileDialog.getOpenFileName(self,"模型路径选择")
        print(model_dir)
        if not isinstance(model_dir,str):
            return
        self.lineEdit_model.setText(model_dir)

    def openfolder_input(self):
        input_dir = QFileDialog.getExistingDirectory(self,"输入图像路径选择")
        print(input_dir)        
        self.lineEdit_input.setText(input_dir)

    def openfolder_output(self):
        output_dir = QFileDialog.getExistingDirectory(self,"输出图像模型路径选择")
        print(output_dir)
        self.lineEdit_output.setText(output_dir)

    def showimage(self, image_name):
        name = image_name.split('/')[-1]
        self.label_image_name.setText(name)
        
        path = os.path.join(self.output_path,image_name)
        img = QtGui.QPixmap(path)
        self.label_image.setPixmap(img)

    def pageup(self):
        if self.output_path == None:
            return
        else:
            image = []
            for image_format in IMAGE_FORMAT:
                image += glob.glob(os.path.join(self.output_path, '*.'+image_format))

        if len(image) == 0:
            QMessageBox.warning(self,"警告","现在还没有输出")
        else:
            self.page -= 1
            self.page = min(max(0,self.page),len(image)-1)
            print(self.page)
            self.showimage(image[self.page])
        
    def pagedown(self):
        if self.output_path == None:
            return
        else:
            image = []
            for image_format in IMAGE_FORMAT:
                image += glob.glob(os.path.join(self.output_path, '*.'+image_format))

        if len(image) == 0:
            QMessageBox.warning(self,"警告","现在还没有输出")
        else:
            self.page += 1
            self.page = min(max(0,self.page),len(image)-1)
            print(self.page)
            self.showimage(image[self.page])
    
    def detect(self):
        
        model_dir = self.lineEdit_model.text()
        input_dir = self.lineEdit_input.text()
        output_dir = self.lineEdit_output.text()
        model_dir = os.path.abspath(model_dir)
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(model_dir):
            QMessageBox.warning(self,"警告","路径"+model_dir+"不存在")
            
            return
        """
        else:
            model_dir = os.path.abspath(model_dir)
        """
        if not os.path.exists(input_dir):
            QMessageBox.warning(self,"警告","路径"+input_dir+"不存在")
            
            return
        """
        else:
            input_dir = os.path.abspath(input_dir)
        """
        if not os.path.exists(output_dir):
            QMessageBox.warning(self,"警告","路径"+output_dir+"不存在")
            
            return
        """
        else:
            output_dir = os.path.abspath(output_dir)
        """
        if os.getcwd() in output_dir:
            QMessageBox.warning(self,"警告","不能选择主程序所在文件夹")
            return
        else:
            QMessageBox.warning(self,"警告","开始检测会清空此文件夹，保留新的检测结果")
            self.output_path = output_dir
            shutil.rmtree(self.output_path)
            os.mkdir(self.output_path)
            self.label_image.clear()
            self.label_image.setText("测试结果")
            self.label_image_name.setText("")
        self.page = 0

        model_index = self.comboBox_model.currentIndex()
        model_name = MODEL[model_index]

        command = " ".join(["bash shells/"+model_name+".sh", model_dir, input_dir, output_dir])
        print(command)
        self.pushButton_start.setEnabled(False)
        self.thread_1 = WorkThread(command)
        self.thread_1._signal.connect(self.set_btn)
        self.thread_1.start()

    def set_btn(self):
        self.pushButton_start.setEnabled(True)
        
    
    def model_SSL_t(self):
        self.comboBox_model.setCurrentIndex(0)
        self.setlineEdits(MODEL[0])
        
    def model_SSL_s(self):
        self.comboBox_model.setCurrentIndex(1)
        self.setlineEdits(MODEL[1])
    """
    def model_SSL_hs(self):
        self.comboBox_model.setCurrentIndex(2)
        self.setlineEdits(MODEL[2])
    """

    def model_mmdet_air_before(self):
        self.comboBox_model.setCurrentIndex(2)
        self.setlineEdits(MODEL[2])

    def model_mmdet_ship_before(self):
        self.comboBox_model.setCurrentIndex(3)
        self.setlineEdits(MODEL[3])
    
    def model_mmdet_air_after(self):
        self.comboBox_model.setCurrentIndex(4)
        self.setlineEdits(MODEL[4])

    def model_mmdet_ship_after(self):
        self.comboBox_model.setCurrentIndex(5)
        self.setlineEdits(MODEL[5])
    
    def model_fewshot_aug(self):
        self.comboBox_model.setCurrentIndex(6)
        self.setlineEdits(MODEL[6])
    
    def model_fewshot_dota(self):
        self.comboBox_model.setCurrentIndex(7)
        self.setlineEdits(MODEL[7])
    
    def model_active_before(self):
        self.comboBox_model.setCurrentIndex(8)
        self.setlineEdits(MODEL[8])

    def model_active_after(self):
        self.comboBox_model.setCurrentIndex(9)
        self.setlineEdits(MODEL[9])
    
    def model_attr(self):
        self.comboBox_model.setCurrentIndex(10)
        self.setlineEdits(MODEL[10])
    
    def setlineEdits(self, model_name):
        tmp = self.config[model_name]
        self.lineEdit_model.setText(tmp['model'])
        self.lineEdit_input.setText(tmp['input'])
        self.lineEdit_output.setText(tmp['output'])
    
    def setupConnect(self,Form):
        self.pushButton_model_dir.clicked.connect(Form.openfolder_model)
        self.pushButton_input_dir.clicked.connect(Form.openfolder_input)
        self.pushButton_output_dir.clicked.connect(Form.openfolder_output)        
        self.pushButton_up.clicked.connect(Form.pageup)
        self.pushButton_down.clicked.connect(Form.pagedown)
        self.pushButton_start.clicked.connect(Form.detect)
        self.pushButton_SSL_t.clicked.connect(Form.model_SSL_t)
        self.pushButton_SSL_s.clicked.connect(Form.model_SSL_s)
        #self.pushButton_SSL_hs.clicked.connect(Form.model_SSL_hs)
        self.pushButton_mmdet_air_before.clicked.connect(Form.model_mmdet_air_before)
        self.pushButton_mmdet_ship_before.clicked.connect(Form.model_mmdet_ship_before)
        self.pushButton_mmdet_air_after.clicked.connect(Form.model_mmdet_air_after)
        self.pushButton_mmdet_ship_after.clicked.connect(Form.model_mmdet_ship_after)
        self.pushButton_fewshot_aug.clicked.connect(Form.model_fewshot_aug)
        self.pushButton_fewshot_dota.clicked.connect(Form.model_fewshot_dota)
        self.pushButton_active_before.clicked.connect(Form.model_active_before)
        self.pushButton_active_after.clicked.connect(Form.model_active_after)
        self.pushButton_attr.clicked.connect(Form.model_attr)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    ui =  TMP()

    ui.show()
    sys.exit(app.exec_())
