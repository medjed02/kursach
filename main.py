#!/usr/bin/python
# -*- coding: UTF-8 -*-

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QObject, QMetaObject, QRect, QSize
from PySide6.QtWidgets import QMainWindow, QFileDialog, QWidget, QLabel, QPushButton
from PIL import Image, ImageEnhance
import argparse
import cv2
import glob
import json
import numpy as np
import os
import sys


def pil2pixmap(im):
    if im.mode == "RGB":
        r, g, b = im.split()
        im = Image.merge("RGB", (r, g, b))
    elif im.mode == "RGBA":
        r, g, b, a = im.split()
        im = Image.merge("RGBA", (b, r, g, b))
    elif im.mode == "L":
        im = im.convert("RGBA")

    im2 = im.convert("RGBA")
    data = im2.tobytes("raw", "RGBA")
    qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_ARGB32)
    pixmap = QtGui.QPixmap.fromImage(qim)
    return pixmap


class UiMainWindow(object):
    def setup_ui(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(600, 650)
        MainWindow.setMinimumSize(QSize(600, 650))
        MainWindow.setMaximumSize(QSize(600, 650))
        MainWindow.setStyleSheet(u"QWidget {\n"
                                 "	color: white;\n"
                                 "	background-color: #222;\n"
                                 "	font-family: Rubik;\n"
                                 "	font-size: 16pt;\n"
                                 "	font-weight: 600;\n"
                                 "}")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.btn_choose_file = QPushButton(self.centralwidget)
        self.btn_choose_file.setObjectName(u"btn_choose_file")
        self.btn_choose_file.setGeometry(QRect(150, 580, 300, 50))
        self.btn_choose_file.setStyleSheet(u"QPushButton {\n"
                                           "	background-color: #333;\n"
                                           "	border: none;\n"
                                           "}\n"
                                           "\n"
                                           "QPushButton:hover {\n"
                                           "	background-color: #666;\n"
                                           "}\n"
                                           "\n"
                                           "QPushButton:pressed {\n"
                                           "	background-color: #888;\n"
                                           "}")
        self.image_widget = QLabel(self.centralwidget)
        self.image_widget.setObjectName(u"image_widget")
        self.image_widget.setGeometry(QRect(40, 40, 520, 520))
        MainWindow.setCentralWidget(self.centralwidget)

        MainWindow.setWindowTitle("Анализатор дефектов")
        self.btn_choose_file.setText("Выбрать файл")
        self.image_widget.setText("")

        QMetaObject.connectSlotsByName(MainWindow)


class ImageLoader:
    def __init__(self, input_name):
        self.input_name = input_name
        self.start_images = []
        self.images = []
        self.filenames = []
        self.files = []

    def load_images(self):
        if os.path.isdir(self.input_name):
            self.files = glob.glob(self.input_name + '/**/*.png', recursive=True)
        elif os.path.isabs(self.input_name):
            self.files = [self.input_name]
        else:
            raise RuntimeError("Ошибка исходных данных: Папки или файла с указанным путём не существует")

        self.filenames = ["result_" + os.path.basename(file) for file in self.files]
        try:
            self.start_images = [np.array(Image.open(file), dtype=np.float64) for file in self.files]
        except Exception as ex:
            raise FileNotFoundError("Ошибка исходных данных: Один из файлов не может быть открыт (или некорректен)")

        # pairs of image and reverse image
        self.images = [(im, 255 - im) for im in self.start_images]

    def upload_images(self, processed_images, output_name):
        if not os.path.isdir(output_name):
            raise RuntimeError("Ошибка исходных данных: Папки или файла с указанным путём не существует")

        if len(processed_images) != len(self.filenames):
            raise RuntimeError("Ошибка выгрузки файлов: Количество изображений "
                               "не совпадает с количеством имен файлов (внутренняя ошибка)")

        for i in range(len(processed_images)):
            filename = os.path.join(output_name, self.filenames[i])
            is_success, im_buf_arr = cv2.imencode(".png", processed_images[i])
            if not is_success:
                raise RuntimeError("Ошибка выгрузки файлов: Проблема кодирования итоговых изображений в PNG формат")
            try:
                im_buf_arr.tofile(filename)
            except Exception as ex:
                raise RuntimeError("Ошибка выгрузки файлов: Проблема кодирования итоговых изображений в PNG формат")

    def print_verdicts(self, verdicts, rectangles):
        if len(verdicts) != len(self.filenames):
            raise RuntimeError("Ошибка вывода вердиктов: количество вердиктов и "
                               "количество входных файлов не совпадают")
        for i in range(len(verdicts)):
            if verdicts[i]:
                print(f"{self.files[i]}: дефект найден")
                for rec in rectangles[i]:
                    print(*rec)
            else:
                print(f"{self.files[i]}: дефект не найден")

    def save_verdicts(self, verdicts, rectangles, json_filename):
        if len(verdicts) != len(self.filenames):
            raise RuntimeError("Ошибка сохранения вердиктов: количество вердиктов и "
                               "количество входных файлов не совпадают")
        json_dict = dict()
        json_dict['Analyzer verdicts'] = list()
        for i in range(len(self.filenames)):
            json_dict['Analyzer verdicts'].append(dict())
            json_dict['Analyzer verdicts'][i]['filename'] = self.filenames[i]
            json_dict['Analyzer verdicts'][i]['defects'] = verdicts[i]
            json_dict['Analyzer verdicts'][i]['blocked areas'] = list()
            for area in rectangles[i]:
                json_dict['Analyzer verdicts'][i]['blocked areas'].append({'x1': area[0], 'y1': area[1],
                                                                           'x2': area[2], 'y2': area[3]})
        try:
            result_filename = os.path.join(json_filename, 'result_verdicts.json')
            with open(result_filename, "w") as write_file:
                json.dump(json_dict, write_file)
        except Exception as ex:
            raise RuntimeError("Ошибка сохранения вердиктов: ошибка сохранения файла "
                               "(вероятно, неправильный путь до папки назначения")


class ImageAnalyzer:
    def __init__(self, images, start_images):
        self.images = images
        self.start_images = start_images
        self.verdicts = []
        self.defect_rectangles = []

    def image_clear_noise(self, image):
        # cutting out the noise frequencies
        image = np.fft.fft2(image)
        mask = np.zeros(image.shape, dtype=bool)
        try:
            mask[0:20, 0:50] = True
            mask[-40:, 0:70] = True
            mask[0:40, -70:] = True
            mask[-20:, -50:] = True
        except Exception as ex:
            raise RuntimeError("Ошибка исходных данных: одно из изображений имеет неподдерживаемый размер")
        image[mask] = 0
        image = np.fft.ifft2(image).real

        # increase contrast for median blur of the noise
        image[image < 0] = 0
        image[image > 255] = 255
        image = Image.fromarray(image.astype('uint8')).convert(mode="L")
        image = ImageEnhance.Contrast(image).enhance(30)
        image = np.array(image, dtype=np.float64)

        try:
            n, m = image.shape
            image[:70, :400] = 0
            image[:250, :50] = 0

            image[n - 250:, :70] = 0
            image[n - 70:, :400] = 0

            image[n - 250:, m - 70:] = 0
            image[n - 70:, m - 400:] = 0

            image[:300, m - 40:] = 0
            image[:150, m - 60:] = 0
            image[:70, m - 400:] = 0

            image[0:20, :] = 0
            image[:, 0:20] = 0
            image[-20:, :] = 0
            image[:, -20:] = 0
        except Exception as ex:
            raise RuntimeError("Ошибка исходных данных: одно из изображений имеет неподдерживаемый размер")

        # median blur of the noise
        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2BGR)
        image = cv2.medianBlur(image, 5)
        return image

    def clear_noise(self):
        # clear noise from images and from reversed images
        for i in range(len(self.images)):
            self.images[i] = (self.image_clear_noise(self.images[i][0]),
                              self.image_clear_noise(self.images[i][1]))

    def find_defects(self):
        for i in range(len(self.images)):
            # highlight borders in image and reversed image and sum results
            self.images[i] = cv2.Canny(self.images[i][0], 150, 190) + cv2.Canny(self.images[i][1], 150, 190)

            # "blur" the boundaries
            new_image = np.zeros(self.images[i].shape, dtype=int)
            for pixel in np.argwhere(self.images[i] == 255):
                for dx in range(-20, 21):
                    for dy in range(-(20 - abs(dx)), 21 - abs(dx)):
                        x = pixel[0] + dx
                        y = pixel[1] + dy
                        if 0 <= x < new_image.shape[0] and 0 <= y < new_image.shape[1]:
                            new_image[x, y] = 255

            # get verdicts
            if len(set(self.images[i].flatten())) > 1:
                self.verdicts.append(True)
            else:
                self.verdicts.append(False)

            # return to start image and increase contrast
            self.images[i] = Image.fromarray(self.start_images[i].astype('uint8')).convert(mode="L")
            self.images[i] = ImageEnhance.Contrast(self.images[i]).enhance(30)
            self.images[i] = np.array(self.images[i], dtype='uint8')
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR)

            # find contours of "blured" boundaries and fit them into rectangles in the contrasted start image
            contours, _ = cv2.findContours(new_image.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            self.defect_rectangles.append([])
            for contour in contours:
                x1, y1, len_x, len_y = cv2.boundingRect(contour)
                if min(len_x, len_y) < 60:
                    size = 120
                else:
                    size = 200
                self.defect_rectangles[i].append((x1 + len_x // 2 - size // 2,
                                                  y1 + len_y // 2 - size // 2,
                                                  x1 + len_x // 2 + size // 2,
                                                  y1 + len_y // 2 + size // 2))
                cv2.rectangle(self.images[i], (self.defect_rectangles[i][-1][0], self.defect_rectangles[i][-1][1]),
                              (self.defect_rectangles[i][-1][2], self.defect_rectangles[i][-1][3]), (0, 0, 255), 3)


class MainWindow(QMainWindow, UiMainWindow):
    def __init__(self):
        UiMainWindow.__init__(self)
        QMainWindow.__init__(self)

        # Initialize UI
        self.setup_ui(self)
        self.btn_choose_file.clicked.connect(self.show_image)

    def tr(self, text):
        return QObject.tr(text)

    def show_image(self):
        path_to_file, _ = QFileDialog.getOpenFileName(self, self.tr("Load Image"), self.tr("~/Desktop/"),
                                                      self.tr("PNG images (*.png)"))

        loader = ImageLoader(path_to_file)
        loader.load_images()

        analyzer = ImageAnalyzer(loader.images, loader.start_images)
        analyzer.clear_noise()
        analyzer.find_defects()

        image = Image.fromarray(analyzer.images[0])
        image = image.resize((self.image_widget.height(), self.image_widget.width()), Image.LANCZOS)
        self.image_widget.setPixmap(pil2pixmap(image))


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--console', action='store_true')
    parser.add_argument('-i', '--input', default=None)
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('-j', '--json', default=None)

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if args.console:
        if args.input is None:
            raise RuntimeError("Ошибка входных данных: данные не указаны")

        loader = ImageLoader(args.input)
        loader.load_images()

        analyzer = ImageAnalyzer(loader.images, loader.start_images)
        analyzer.clear_noise()
        analyzer.find_defects()

        loader.print_verdicts(analyzer.verdicts, analyzer.defect_rectangles)

        if args.output:
            loader.upload_images(analyzer.images, args.output)

        if args.json:
            loader.save_verdicts(analyzer.verdicts, analyzer.defect_rectangles, args.json)
    else:
        app = QtWidgets.QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec())
