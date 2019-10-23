import enum
import json
import os
import shutil
import tkinter as tk
from datetime import datetime
from functools import partial
from tkinter import ttk

from skimage.io import imsave

from .partially_labelled_dataset import (
    PartiallyLabelledDataset, create_rgb_mask,
    create_class_mask, create_instance_mask
)
from ..base import (
    get_rect, verify_or_create_directory, random_colors
)


class ExportType(enum.Enum):
    PascalVOC = 0
    COCO = 1
    KITTI = 2
    NLABJSON = 3


class XmlWriter:
    def __init__(self, fpath):
        self.fp = open(fpath, 'w')
        self.indent = 0

    def write(self, content):
        self.fp.write('\t' * self.indent + content + '\n')

    def open_block(self, block_name):
        self.write('<{}>'.format(block_name))
        self.indent += 1

    def close_block(self, block_name):
        self.indent -= 1
        self.write('</{}>'.format(block_name))

    def add_item(self, item_name, value):
        self.write('<{}>{}</{}>'.format(item_name, value, item_name))

    def __del__(self):
        self.fp.close()


def write_pascal_voc(path, image_name, image_rect, annotation, class_id2name):
    writer = XmlWriter(path)

    writer.open_block('annotation')

    writer.add_item('filename', image_name)
    writer.add_item('path', 'images/' + image_name)

    writer.open_block('source')
    writer.add_item('database', 'Nearthlab Inc.')
    writer.close_block('source')

    writer.open_block('size')
    writer.add_item('width', image_rect.width())
    writer.add_item('height', image_rect.height())
    writer.add_item('depth', 3)
    writer.close_block('size')

    writer.add_item('segmented', 0)

    for obj_idx, bbox in enumerate([anno.bbox.intersect(image_rect) for anno in annotation]):
        writer.open_block('object')

        writer.add_item('name', class_id2name[annotation[obj_idx].class_id])
        writer.add_item('pose', 'unspecified')
        writer.add_item('truncated', 0)
        writer.add_item('difficult', 0)
        writer.add_item('occluded', 0)

        writer.open_block('bndbox')
        writer.add_item('xmin', bbox.left)
        writer.add_item('ymin', bbox.top)
        writer.add_item('xmax', bbox.right)
        writer.add_item('ymax', bbox.bottom)
        writer.close_block('bndbox')

        writer.close_block('object')
    writer.close_block('annotation')


class ExportHelper:
    def __init__(self, dataset: PartiallyLabelledDataset, cx=None, cy=None):
        self.root = tk.Tk()
        self.root.title('Export Options')
        w, h = 400, 350
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

        # Export type
        self.export_type = ExportType(0)

        # Size of the validation set
        self.num_val = 0

        # Export type
        tk.Label(self.root, text='Export Type', wraplength=4 * w // 5).pack(pady=(25, 0))
        options = ExportType.__members__.keys()
        self.export_type_buttons = []
        for i, option in enumerate(options):
            button = tk.Radiobutton(
                self.root, text=option, value=i,
                command=partial(self.set_export_type, v=i)
            )
            if i == 0:
                button.select()
            button.pack(anchor='w')
            self.export_type_buttons.append(button)

        # Size of the validation set
        self.nv_label = tk.Label(self.root, text=self.scale_text)
        self.nv_label.pack(pady=(15, 0))
        unit = max(self.dataset.num_images // 5, 1) // 10 * 10
        self.nv_scale = tk.Scale(
            self.root, command=self.set_num_val,
            orient='horizontal', showvalue=False,
            tickinterval=unit, to=self.dataset.num_images,
            length=300
        )
        self.nv_scale.pack()

        self.progress_label = tk.Label(self.root, text='Press "Export" when ready')
        self.progress_label.pack(pady=(15, 0))
        self.progress_var = tk.DoubleVar()
        self.progressbar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100, length=300)
        self.progressbar.pack()

        self.export_button = tk.Button(self.root, text='Export', command=self.export)
        self.export_button.pack(pady=(20, 0))
        self.quit_button = tk.Button(self.root, text='Quit', command=self.close)
        self.quit_button.pack(pady=(5, 0))
        self.root.bind('<Return>', lambda x: self.export())
        self.root.bind('<Escape>', lambda x: self.close())
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    @property
    def scale_text(self):
        return 'size of the validation set: {}'.format(self.num_val)

    def set_export_type(self, v):
        self.export_type = ExportType(v)

    def set_num_val(self, v):
        self.num_val = int(v)
        self.nv_label.config(text=self.scale_text)

    def copy_image_files(self, dst_dir, dataset: PartiallyLabelledDataset, subset):
        for i, image_path in enumerate(dataset.image_files):
            percentage = i / dataset.num_images * 100
            self.progress_var.set(percentage)
            self.progress_label.config(text='[%s] Copying image files (%.1f%%)...' % (subset, percentage))
            self.progressbar.update()
            shutil.copy(image_path, os.path.join(dst_dir, os.path.basename(image_path)))

    def export_pascal_voc(self, base_dir, dataset: PartiallyLabelledDataset, subset):
        image_dir = os.path.join(base_dir, 'images')
        verify_or_create_directory(image_dir)
        self.copy_image_files(image_dir, dataset, subset)
        obj_dir = os.path.join(base_dir, 'objects')
        verify_or_create_directory(obj_dir)

        for i in range(dataset.num_images):
            percentage = i / dataset.num_images * 100
            self.progress_var.set(percentage)
            self.progress_label.config(text='[%s] Writing PASCAL VOC annotation files (%.1f%%)...' % (subset, percentage))
            self.progressbar.update()
            image_fullname = dataset.image_files[i]
            image_name = os.path.basename(image_fullname)
            image_rect = get_rect(dataset.load_image(i))
            annotation = dataset.load_annotations(i)

            xml_path = os.path.join(obj_dir, image_name[:-4] + '.xml')
            write_pascal_voc(xml_path, image_name, image_rect, annotation, dataset.class_id2name)

    def export_coco(self, base_dir, dataset: PartiallyLabelledDataset, subset):
        if dataset.num_images == 0:
            return

        date = datetime.now()
        year = str(date.year)

        anno_dir = os.path.join(base_dir, 'annotations')
        verify_or_create_directory(anno_dir)

        json_path = os.path.join(anno_dir, 'instances_{}{}.json'.format(subset, year))

        image_dir = os.path.join(base_dir, subset + year)
        verify_or_create_directory(image_dir)

        header = {
            'description': 'Blade & defect segmentation dataset created by Nearthlab',
            'url': 'http://nearthlab.com/en',
            'version': date.strftime('%Y%m%d'),
            'year': year,
            'contributor': 'Nearthlab Inc.',
            'date_created': date.strftime('%Y-%m-%d %H:%M:%S.%f')
        }

        categories = [
            {
                'supercategory': dataset.class_id2name[class_id],
                'id': class_id,
                'name': dataset.class_id2name[class_id]
            } for class_id in range(dataset.num_classes)
        ]

        images = []
        annotations = []
        annotation_id = 1
        for image_id in range(dataset.num_images):
            percentage = image_id / dataset.num_images * 100
            self.progress_var.set(percentage)
            self.progress_label.config(text='[%s] Creating COCO annotation (%.1f%%)...' % (subset, percentage))
            self.progressbar.update()

            image_fullname = dataset.image_files[image_id]
            image_name = os.path.basename(image_fullname)
            img = dataset.load_image(image_id)
            image_rect = get_rect(img)
            images.append({
                'license': 1,
                'file_name': image_name,
                'coco_url': 'N/A',
                'height': img.shape[0],
                'width': img.shape[1],
                'date_captured': 'N/A',
                'flickr_url': 'N/A',
                'id': image_id + 1
            })

            for anno in dataset.load_annotations(image_id):
                annotations.append({
                    'segmentation': anno.coco_json(),
                    'iscrowd': 0,
                    'image_id': image_id + 1,
                    'category_id': anno.class_id,
                    'id': annotation_id,
                    'bbox': list(anno.bbox.intersect(image_rect)),
                    'area': sum([poly.area for poly in anno.polys])
                })
                annotation_id += 1

        self.progress_var.set(100)
        self.progress_label.config(text='Successfully created COCO annotation!')
        self.progressbar.update()

        with open(json_path, 'w') as fp:
            json.dump({
                'info': header,
                'images': images,
                'annotations': annotations,
                'categories': categories
            }, fp, sort_keys=True)

        self.copy_image_files(image_dir, dataset, subset)

    def export_kitti(self, base_dir, dataset: PartiallyLabelledDataset, subset):
        if dataset.num_images == 0:
            return

        anno_dir = os.path.join(base_dir, '..', 'annotations')
        verify_or_create_directory(anno_dir)
        pallete = random_colors(dataset.num_classes, bright=False, seed=6, uint8=True)

        label = [{
            'name': 'background', 'id': 0, 'trainId': 255,
            'category': 'background', 'catId': 0, 'hasInstances': False,
            'ignoreInEval': True, 'color': (0, 0, 0)
        }]
        for class_id in range(dataset.num_classes):
            class_name = dataset.class_id2name[class_id]
            color = tuple([val.item() for val in pallete[class_id]])
            label.append({
                'name': class_name, 'id': class_id + 1, 'trainId': class_id,
                'category': class_name, 'catId': class_id + 1, 'hasInstances': True,
                'ignoreInEval': False, 'color': color
            })

        with open(
                os.path.join(
                    anno_dir,
                    'semantic_{}.json'.format(datetime.now().year)
                ), 'w'
        ) as fp:
            json.dump(label, fp, sort_keys=True)

        image_dir = os.path.join(base_dir, 'images')
        verify_or_create_directory(image_dir)
        self.copy_image_files(image_dir, dataset, subset)

        sem_rgb_dir = os.path.join(base_dir, 'semantic_rgb')
        sem_dir = os.path.join(base_dir, 'semantic')
        inst_dir = os.path.join(base_dir, 'instance')
        verify_or_create_directory(sem_rgb_dir)
        verify_or_create_directory(sem_dir)
        verify_or_create_directory(inst_dir)

        for image_id in range(dataset.num_images):
            percentage = image_id / dataset.num_images * 100
            self.progress_var.set(percentage)
            self.progress_label.config(text='[%s] Writing KITTI mask images (%.1f%%)...' % (subset, percentage))
            self.progressbar.update()

            shape = dataset.load_image(image_id).shape
            image_fullname = dataset.image_files[image_id]
            image_name = os.path.basename(image_fullname)

            annotations = dataset.load_annotations(image_id)
            rgb_mask = create_rgb_mask(annotations, pallete, shape)
            cls_mask = create_class_mask(annotations, shape)
            inst_mask = create_instance_mask(annotations, shape)
            imsave(
                os.path.join(
                    sem_rgb_dir,
                    image_name[:-4] + '.png'
                ),
                rgb_mask,
                check_contrast=False
            )
            imsave(
                os.path.join(
                    sem_dir,
                    image_name[:-4] + '.png'
                ),
                cls_mask,
                check_contrast=False
            )
            imsave(
                os.path.join(
                    inst_dir,
                    image_name[:-4] + '.png'
                ),
                inst_mask,
                check_contrast=False
            )

    def export_nlabjson(self, base_dir, dataset: PartiallyLabelledDataset, subset):
        if dataset.num_images == 0:
            return
        image_dir = os.path.join(base_dir, 'images')
        verify_or_create_directory(image_dir)
        self.copy_image_files(image_dir, dataset, subset)

        anno_dir = os.path.join(base_dir, 'annotations')
        verify_or_create_directory(anno_dir)

        for image_id in range(dataset.num_images):
            percentage = image_id / dataset.num_images * 100
            self.progress_var.set(percentage)
            self.progress_label.config(text='[%s] Writing Nearthlab json annotations (%.1f%%)...' % (subset, percentage))
            self.progressbar.update()

            image = dataset.load_image(image_id)
            image_shape = image.shape
            image_rect = get_rect(image)
            image_fullname = dataset.image_files[image_id]
            image_name = os.path.basename(image_fullname)

            annotations = dataset.load_annotations(image_id)
            shapes = []
            for a in annotations:
                bbox = a.bbox.intersect(image_rect)
                shapes.append({
                    'label': dataset.class_id2name[a.class_id],
                    'points': [
                        list(bbox.tl_corner),
                        list(bbox.br_corner)
                    ]
                })

            with open(
                    os.path.join(
                        anno_dir,
                        image_name[:-4] + '.json'
                    ), 'w'
            ) as fp:
                json.dump({
                    'imageHeight': image_shape[0],
                    'imageWidth': image_shape[1],
                    'imagePath': image_name,
                    'shapes': shapes
                }, fp, sort_keys=True, indent=2)

    def export(self):
        self.export_button.config(state=tk.DISABLED)
        self.nv_scale.config(state=tk.DISABLED)
        for button in self.export_type_buttons:
            button.config(state=tk.DISABLED)

        train_dataset, val_dataset = self.dataset.split_train_val(self.num_val)

        exported_dir = os.path.join(self.dataset.root, 'exported')
        verify_or_create_directory(exported_dir)
        target_dir = os.path.join(exported_dir, self.export_type.name)
        verify_or_create_directory(target_dir)

        train_dir = os.path.join(target_dir, 'train')
        val_dir = os.path.join(target_dir, 'val')

        if self.export_type == ExportType.PascalVOC:
            verify_or_create_directory(train_dir)
            verify_or_create_directory(val_dir)
            self.export_pascal_voc(train_dir, train_dataset, 'train')
            self.export_pascal_voc(val_dir, val_dataset, 'val')
        elif self.export_type == ExportType.COCO:
            self.export_coco(target_dir, train_dataset, 'train')
            self.export_coco(target_dir, val_dataset, 'val')
        elif self.export_type == ExportType.KITTI:
            verify_or_create_directory(train_dir)
            verify_or_create_directory(val_dir)
            self.export_kitti(train_dir, train_dataset, 'train')
            self.export_kitti(val_dir, val_dataset, 'val')
        else:
            verify_or_create_directory(train_dir)
            verify_or_create_directory(val_dir)
            self.export_nlabjson(train_dir, train_dataset, 'train')
            self.export_nlabjson(val_dir, val_dataset, 'val')

        self.result = target_dir
        self.close()

    def close(self):
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
        return self.result
