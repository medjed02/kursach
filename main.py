#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import glob
import os
import argparse
import cv2
from PIL import Image


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
            filename = self.output_name + "\\" + self.filenames[i]
            cv2.imwrite(filename, processed_images[i])


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