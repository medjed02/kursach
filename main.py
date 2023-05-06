#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import glob
import os
import argparse
import cv2
from PIL import Image, ImageEnhance


class ImageLoader:
    def __init__(self, input_name, output_name):
        self.input_name = input_name
        self.output_name = output_name
        self.images = []
        self.filenames = []

    def load_images(self):
        if os.path.isdir(self.input_name):
            files = glob.glob(self.input_name + '/**/*.png', recursive=True)
        elif os.path.isabs(self.input_name):
            files = [self.input_name]
        self.filenames = ["result_" + file.split('\\')[-1] for file in files]
        self.images = [np.array(Image.open(file), dtype=np.float64) for file in files]

    def upload_images(self, processed_images):
        # TODO len(processed_images) != len(self.filenames)
        for i in range(len(processed_images)):
            filename = os.path.join(self.output_name, self.filenames[i])
            print(cv2.imwrite(filename, processed_images[i]))


class ImageAnalyzer:
    def __init__(self, images):
        self.images = images

    def clear_noise(self):
        for i in range(len(self.images)):
            # cutting out the noise frequencies
            self.images[i] = np.fft.fft2(self.images[i])
            mask = np.zeros(self.images[i].shape, dtype=bool)
            mask[0:20, 0:50] = True
            mask[-40:, 0:70] = True
            mask[0:40, -70:] = True
            mask[-20:, -50:] = True
            self.images[i][mask] = 0
            self.images[i] = np.fft.ifft2(self.images[i]).real

            # increase contrast for median blur of the noise
            self.images[i] = Image.fromarray(self.images[i].astype(int)).convert(mode="L")
            self.images[i] = ImageEnhance.Contrast(self.images[i]).enhance(30)
            self.images[i] = np.array(self.images[i], dtype=np.float64)
            self.images[i][0:60, :] = 0
            self.images[i][:, 0:60] = 0
            self.images[i][-60:, :] = 0
            self.images[i][:, -60:] = 0

            # median blur of the noise
            self.images[i] = cv2.cvtColor(self.images[i].astype('uint8'), cv2.COLOR_GRAY2BGR)
            self.images[i] = cv2.medianBlur(self.images[i], 5)

    def find_defects(self):
        for i in range(len(self.images)):
            self.images[i] = cv2.Canny(self.images[i], 150, 190)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?')
    parser.add_argument('output', nargs='?')

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    loader = ImageLoader(args.input, args.output)
    loader.load_images()

    analyzer = ImageAnalyzer(loader.images)
    analyzer.clear_noise()
    analyzer.find_defects()

    loader.upload_images(analyzer.images)
