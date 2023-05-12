#!/usr/bin/python
# -*- coding: UTF-8 -*-

from PIL import Image, ImageEnhance
import argparse
import cv2
import glob
import numpy as np
import os


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
        self.start_images = [np.array(Image.open(file), dtype=np.float64) for file in self.files]

        # pairs of image and reverse image
        self.images = [(im, 255 - im) for im in self.start_images]

    def upload_images(self, processed_images, output_name):
        if not os.path.isdir(output_name):
            raise RuntimeError("Ошибка исходных данных: Папки или файла с указанным путём не существует")

        if len(processed_images) != len(self.filenames):
            raise RuntimeError("Error of uploading images: cnt of images is not equal cnt of filenames")

        for i in range(len(processed_images)):
            filename = os.path.join(output_name, self.filenames[i])
            is_success, im_buf_arr = cv2.imencode(".png", processed_images[i])
            if not is_success:
                raise RuntimeError("Error of uploading images: encoding error")
            im_buf_arr.tofile(filename)

    def print_verdicts(self, verdicts, rectangles):
        if len(verdicts) != len(self.filenames):
            raise RuntimeError("Print verdicts error: cnt of verdicts is not equal cnt of filenames")
        for i in range(len(verdicts)):
            if verdicts[i]:
                print(f"{self.files[i]}: дефект найден")
                for rec in rectangles:
                    print(*rec)
            else:
                print(f"{self.files[i]}: дефект не найден")


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
        mask[0:20, 0:50] = True
        mask[-40:, 0:70] = True
        mask[0:40, -70:] = True
        mask[-20:, -50:] = True
        image[mask] = 0
        image = np.fft.ifft2(image).real

        # increase contrast for median blur of the noise
        image = Image.fromarray(image.astype(int)).convert(mode="L")
        image = ImageEnhance.Contrast(image).enhance(30)
        image = np.array(image, dtype=np.float64)
        image[0:60, :] = 0
        image[:, 0:60] = 0
        image[-60:, :] = 0
        image[:, -60:] = 0

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

            # return to start image and increase contrast
            self.images[i] = Image.fromarray(self.start_images[i].astype(int)).convert(mode="L")
            self.images[i] = ImageEnhance.Contrast(self.images[i]).enhance(30)
            self.images[i] = np.array(self.images[i], dtype='uint8')
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR)

            # find contours of "blured" boundaries and fit them into rectangles in the contrasted start image
            contours, _ = cv2.findContours(new_image.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x1, y1, len_x, len_y = cv2.boundingRect(contour)
                if min(len_x, len_y) < 60:
                    size = 120
                else:
                    size = 200
                self.defect_rectangles.append((x1 + len_x // 2 - size // 2,
                                               y1 + len_y // 2 - size // 2,
                                               x1 + len_x // 2 + size // 2,
                                               y1 + len_y // 2 + size // 2))
                cv2.rectangle(self.images[i], (self.defect_rectangles[-1][0], self.defect_rectangles[-1][1]),
                              (self.defect_rectangles[-1][2], self.defect_rectangles[-1][3]), (0, 0, 255), 3)

            # get verdicts
            if len(set(self.images[i].flatten())) > 1:
                self.verdicts.append(True)
            else:
                self.verdicts.append(False)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?')
    parser.add_argument('-o', '--output', default=None)

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    loader = ImageLoader(args.input)
    loader.load_images()

    analyzer = ImageAnalyzer(loader.images, loader.start_images)
    analyzer.clear_noise()
    analyzer.find_defects()

    loader.print_verdicts(analyzer.verdicts, analyzer.defect_rectangles)

    if args.output:
        loader.upload_images(analyzer.images, args.output)
