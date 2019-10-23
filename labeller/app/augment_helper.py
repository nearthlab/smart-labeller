import json
import os
import random
import shutil
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, ttk

import numpy as np
from skimage.io import imsave
from skimage.transform import resize

from .partially_labelled_dataset import (
    PartiallyLabelledDataset,
    save_annotations,
    flip_annotations
)
from ..base import (
    get_rect, Rectangle,
    verify_or_create_directory,
    get_files_in_directory_tree,
    load_rgb_image
)


def crop_image(img: np.ndarray, rect: Rectangle):
    roi = rect.intersect(get_rect(img))
    if roi.is_empty():
        raise ValueError('{} does not overlap with the image region'.format(rect))
    return img[roi.top:roi.bottom + 1, roi.left:rect.right + 1, :]


def get_random_subregion(parent: Rectangle, w, h):
    if parent.width() < w or parent.height() < h or w <= 0 or h <= 0:
        print(parent.width() < w, parent.height() < h)
        raise ValueError('Invalid input given: parent: {}, w: {}, h: {}'.format(parent, w, h))

    left = random.randint(0, parent.width() - w)
    top = random.randint(0, parent.height() - h)

    right = left + w - 1
    bottom = top + h - 1

    return Rectangle(left, top, right, bottom)


def replace_background(img: np.ndarray, bgimg: np.ndarray, mask: np.ndarray):
    assert img.ndim == 3 and bgimg.ndim == 3 and mask.ndim == 2 and mask.dtype == bool
    img_rect = get_rect(img)
    bg_rect = get_rect(bgimg)
    if img_rect not in bg_rect:
        rate = max(img_rect.width() / bg_rect.width(), img_rect.height() / bg_rect.height())
        dst_width = int(round(rate * bg_rect.width()))
        dst_height = int(round(rate * bg_rect.height()))
        bgimg = resize(bgimg, (dst_height, dst_width), preserve_range=True)
        bg_rect = get_rect(bgimg)

    bgimg = crop_image(
        bgimg, get_random_subregion(bg_rect, img_rect.width(), img_rect.height())
    )

    return (img * np.stack((mask,) * 3, -1) + bgimg * np.stack((1 - mask,) * 3, -1)).astype(np.uint8)


class AugmentHelper:
    def __init__(self, dataset: PartiallyLabelledDataset, cx=None, cy=None):
        self.root = tk.Tk()
        self.root.title('Augmentation Options')
        w, h = 400, 450
        cx = cx or self.root.winfo_screenwidth() // 2
        cy = cy or self.root.winfo_screenheight() // 2
        self.root.geometry('{}x{}+{}+{}'.format(
            w, h,
            cx - w // 2,
            cy - h // 2
        ))

        assert dataset.is_complete
        self.dataset = dataset
        self.result = None

        # Augmentation options
        self.ud_flip = False
        self.lr_flip = False
        self.bg_replace = 0
        self.rand_pix_tform = 0

        # up-down flip
        self.ud_button = tk.Checkbutton(self.root, text='Add up-down flips', command=self.set_ud_flip)
        self.ud_button.pack(pady=(25, 0))

        # left-right flip
        self.lr_button = tk.Checkbutton(self.root, text='Add left-right flips', command=self.set_lr_flip)
        self.lr_button.pack(pady=(15, 0))

        # Background replacement
        self.bg_label = tk.Label(self.root, text=self.bg_text)
        self.bg_label.pack(pady=(15, 0))
        self.bg_scale = tk.Scale(
            self.root, command=self.set_bg_replace,
            orient='horizontal', showvalue=False,
            tickinterval=1, to=10,
            length=300
        )
        self.bg_scale.pack()
        self.bgimg_paths = []
        self.bg_dir_label = tk.Label(self.root, text='Choose a directory containing background images')
        self.bg_dir_label.pack(pady=(15, 0))
        self.bg_dir_button = tk.Button(self.root, overrelief='solid', text='...', width=3, command=lambda: self.load_bgimg_paths())
        self.bg_dir_button.pack(pady=(15, 0))

        # Dataset size
        self.size_label = tk.Label(self.root, text=self.size_text)
        self.size_label.pack(pady=(15, 0))

        self.progress_label = tk.Label(self.root, text='Press "Augment" when ready')
        self.progress_label.pack(pady=(15, 0))
        self.progress_var = tk.DoubleVar()
        self.progressbar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100, length=300)
        self.progressbar.pack()

        self.augment_button = tk.Button(self.root, text='Augment', command=self.augment)
        self.augment_button.pack(pady=(20, 0))
        self.quit_button = tk.Button(self.root, text='Quit', command=self.close)
        self.quit_button.pack(pady=(5, 0))
        self.root.bind('<Return>', lambda x: self.augment())
        self.root.bind('<Escape>', lambda x: self.close())
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    @property
    def bg_text(self):
        return 'number of background replacements per image: {}'.format(self.bg_replace)

    @property
    def size_text(self):
        size = self.dataset.num_images
        if self.ud_flip:
            size *= 2
        if self.lr_flip:
            size *= 2
        if len(self.bgimg_paths) > 0:
            size += size * self.bg_replace
        return 'Augmented dataset size: {}'.format(size)

    def load_bgimg_paths(self):
        dirname = filedialog.askdirectory(
            parent=self.root,
            initialdir='~',
            title='Select a dataset root directory'
        )

        if dirname is not None:
            self.bgimg_paths = get_files_in_directory_tree(dirname, ['.jpg', '.png', '.bmp'])
            self.bg_dir_label.config(text='{} background images found in\n{}'.format(len(self.bgimg_paths), dirname))
            self.size_label.config(text=self.size_text)

    def set_bg_replace(self, v):
        self.bg_replace = int(v)
        self.bg_label.config(text=self.bg_text)
        self.size_label.config(text=self.size_text)

    def set_ud_flip(self):
        self.ud_flip = not self.ud_flip
        self.size_label.config(text=self.size_text)

    def set_lr_flip(self):
        self.lr_flip = not self.lr_flip
        self.size_label.config(text=self.size_text)

    def copy_dataset(self, dataset, new_root):
        if new_root == dataset.root:
            return dataset
        elif os.path.exists(new_root):
            raise FileExistsError('{} already exists. Please choose another name'.format(new_root))

        verify_or_create_directory(new_root)

        image_dir = os.path.join(new_root, 'images')
        label_dir = os.path.join(new_root, 'objects')
        verify_or_create_directory(image_dir)
        verify_or_create_directory(label_dir)

        for image_id in range(dataset.num_images):
            percentage = image_id / dataset.num_images * 100
            self.progress_var.set(percentage)
            self.progress_label.config(text='Copying original dataset (%.1f%%)...' % (percentage))
            self.progressbar.update()

            # Copy the original image & label
            shutil.copy(
                dataset.image_files[image_id],
                os.path.join(
                    image_dir,
                    os.path.basename(dataset.image_files[image_id])
                )
            )
            shutil.copy(
                dataset.infer_label_path(image_id),
                os.path.join(
                    label_dir,
                    os.path.basename(dataset.infer_label_path(image_id))
                )
            )

        with open(os.path.join(new_root, 'class_names.json'), 'w') as fp:
            json.dump(dataset.class_id2name, fp)

        clone = PartiallyLabelledDataset()
        clone.load(new_root)
        return clone

    def add_flip(self, dataset: PartiallyLabelledDataset, orient):
        assert orient in ['lr', 'ud']
        image_flipper = np.fliplr if orient == 'lr' else np.flipud
        shape_idx = 1 if orient == 'lr' else 0
        for image_id in range(dataset.num_images):
            percentage = image_id / dataset.num_images * 100
            self.progress_var.set(percentage)
            self.progress_label.config(text='Adding %s flips (%.1f%%)...' %
                                            ('left-right' if orient == 'lr' else 'up-down', percentage))
            self.progressbar.update()
            img = dataset.load_image(image_id)
            annotations = dataset.load_annotations(image_id)

            image_name = os.path.basename(dataset.image_files[image_id])[:-4] + '_{}.jpg'.format(orient)
            label_name = os.path.basename(dataset.infer_label_path(image_id))[:-5] + '_{}.json'.format(orient)

            # Add flipped image & label
            imsave(
                os.path.join(
                    dataset.image_dir,
                    image_name
                ),
                image_flipper(img)
            )
            save_annotations(
                os.path.join(
                    dataset.label_dir,
                    label_name
                ),
                flip_annotations(annotations, img.shape[shape_idx], orient)
            )

        added = PartiallyLabelledDataset()
        added.load(dataset.root)

        return added

    def add_bg_replace(self, dataset: PartiallyLabelledDataset, bgimg_paths, count):
        if count > 0 and len(bgimg_paths) > 0:
            for image_id in range(dataset.num_images):
                percentage = image_id / dataset.num_images * 100
                self.progress_var.set(percentage)
                self.progress_label.config(text='Adding background replaced images (%.1f%%)...' % (percentage))
                self.progressbar.update()
                img = dataset.load_image(image_id)
                annotations = dataset.load_annotations(image_id)

                mask_shape = img.shape[:2]
                fg_mask = np.zeros(mask_shape, dtype=np.bool)
                for a in annotations:
                    fg_mask = np.bitwise_or(fg_mask, a.mask(mask_shape) > 0)

                base_image_name = os.path.basename(dataset.image_files[image_id])
                base_label_name = os.path.basename(dataset.infer_label_path(image_id))

                for i in range(count):
                    image_name = base_image_name[:-4] + '_bg{}.jpg'.format(i + 1)
                    label_name = base_label_name[:-5] + '_bg{}.json'.format(i + 1)
                    bgimg = load_rgb_image(random.choice(bgimg_paths))

                    # Add flipped image & label
                    imsave(
                        os.path.join(
                            dataset.image_dir,
                            image_name
                        ),
                        replace_background(img, bgimg, fg_mask)
                    )
                    shutil.copy(
                        dataset.infer_label_path(image_id),
                        os.path.join(
                            dataset.label_dir,
                            label_name
                        )
                    )

        added = PartiallyLabelledDataset()
        added.load(dataset.root)

        return added

    def augment(self):
        self.augment_button.config(state=tk.DISABLED)
        self.ud_button.config(state=tk.DISABLED)
        self.lr_button.config(state=tk.DISABLED)
        self.bg_scale.config(state=tk.DISABLED)
        self.bg_dir_button.config(state=tk.DISABLED)

        new_root = self.dataset.root + '_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        clone = self.copy_dataset(self.dataset, new_root)

        if self.lr_flip:
            clone = self.add_flip(clone, orient='lr')

        if self.ud_flip:
            clone = self.add_flip(clone, orient='ud')

        clone = self.add_bg_replace(clone, self.bgimg_paths, self.bg_replace)
        self.result = clone.root
        self.close()

    def close(self):
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
        return self.result
