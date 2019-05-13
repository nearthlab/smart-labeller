import copy
import json
import os
import random

import numpy as np

from .utils import verify_or_create_directory, load_rgb_image
from .geometry import (
    polygons_to_mask, mask_to_polygons,
    flip_polygon, Polygon, extract_bbox_multi
)


class ObjectAnnotation:
    def __init__(self, *args):
        if len(args) == 0:
            self.polys = []
            self.class_id = -1
        elif len(args) == 1:
            if type(args[0]) == str:
                if os.path.isfile(args[0]):
                    with open(args[0], 'r') as fp:
                        d = json.load(fp)
                else:
                    d = json.loads(args[0])
                self.class_id = d['class_id']
                self.polys = [Polygon(x) for x in d['annotation']]
            else:
                raise TypeError('Single argument\'s type must be str')
        elif len(args) == 2:
            if type(args[0]) == Polygon:
                self.polys = [args[0]]
            elif type(args[0]) == list:
                for item in args[0]:
                    assert type(item) == Polygon, 'list contains non-polygon objects'
                self.polys = args[0]
            elif type(args[0]) == np.ndarray:
                self.polys = mask_to_polygons(args[0])
            else:
                raise TypeError('args[0] of unexpected type {}'.format(type(args[0])))

            if type(args[1]) == int:
                self.class_id = args[1]
            else:
                raise TypeError('args[1] of unexpected type {}'.format(type(args[1])))
        else:
            raise Exception('Too many arguments')

    def json(self):
        return {
            'class_id': self.class_id,
            'annotation': [poly.to_ndarray().tolist() for poly in self.polys]
        }

    def coco_json(self):
        return [poly.to_ndarray().flatten().tolist() for poly in self.polys]

    def __str__(self):
        return json.dumps(self.json())

    @property
    def bbox(self):
        return extract_bbox_multi(self.polys)

    def mask(self, shape):
        return polygons_to_mask(self.polys, shape)

    def is_empty(self):
        return len(self.polys) == 0 or self.class_id == -1


def load_annotations(file_path):
    with open(file_path, 'r') as fp:
        raw_annos = json.load(fp)

    annos = []
    for raw_anno in raw_annos:
        annos.append(ObjectAnnotation(json.dumps(raw_anno)))
    return annos


def save_annotations(file_path, annotations: list):
    with open(file_path, 'w') as fp:
        json.dump(
            [annotation.json() for annotation in annotations],
            fp, sort_keys=True
        )

def flip_annotation(annotation: ObjectAnnotation, span, orient):
    return ObjectAnnotation(
        [flip_polygon(poly, span, orient) for poly in annotation.polys],
        annotation.class_id
    )

def flip_annotations(annotations: list, span, orient):
    return [flip_annotation(a, span, orient) for a in annotations]

def create_rgb_mask(annos, pallete, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for anno in annos:
        mask[anno.mask(shape[:2]) != 0] = pallete[anno.class_id]
    return mask

def create_class_mask(annos, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for anno in annos:
        mask[anno.mask(shape[:2]) != 0] = anno.class_id
    return mask

def create_instance_mask(annos, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for obj_id, anno in enumerate(annos):
        mask[anno.mask(shape[:2]) != 0] = obj_id + 1
    return mask


class PartiallyLabelledDataset:
    def __init__(self):
        self.root = None
        self.image_files = []
        self.class_id2name = []
        self.class_name2id = []

    @property
    def num_images(self):
        return len(self.image_files)

    @property
    def num_classes(self):
        return len(self.class_id2name)

    @property
    def is_complete(self):
        for image_id in range(len(self.image_files)):
            if not os.path.isfile(self.infer_label_path(image_id)):
                return False
        return True

    def __len__(self):
        return self.num_images

    def load(self, dataset_dir, labelled_only=False):
        if not os.path.isdir(dataset_dir):
            raise Exception('No such directory: {}'.format(dataset_dir))
        self.root = dataset_dir

        self.image_dir = os.path.join(self.root, 'images')
        self.label_dir = os.path.join(self.root, 'objects')

        if not os.path.isdir(self.image_dir):
            raise Exception('A data set root directory must contain a subdirectory named "images"')
        verify_or_create_directory(self.label_dir)

        self.image_files = [os.path.join(self.image_dir, filename) \
                            for filename in sorted(os.listdir(self.image_dir)) \
                            if filename.lower()[-4:] in ['.jpg', '.png', '.bmp']]
        if labelled_only:
            self.image_files = [path for image_id, path in enumerate(self.image_files) if os.path.isfile(self.infer_label_path(image_id))]
        if self.num_images == 0:
            if labelled_only:
                raise Exception('No labelled image found in the image directory: {}'.format(self.image_dir))
            else:
                raise Exception('No image found in the image directory: {}'.format(self.image_dir))

        class_label_path = os.path.join(self.root, 'class_names.json')
        if not os.path.isfile(class_label_path):
            raise Exception('Could not find class_names.json file in the directory: {}'.format(self.root))
        with open(class_label_path, 'r') as fp:
            self.class_id2name = json.load(fp)
            self.class_name2id = {name: id for id, name in enumerate(self.class_id2name)}

    def infer_label_path(self, id):
        return os.path.join(self.label_dir, os.path.basename(self.image_files[id])[:-4] + '.json')

    def load_image(self, id):
        return load_rgb_image(self.image_files[id])

    def load_annotations(self, id):
        label_path = self.infer_label_path(id)
        if os.path.isfile(label_path):
            return load_annotations(label_path)
        else:
            return []

    def split_train_val(self, num_val, shuffle=True):
        image_files = copy.deepcopy(self.image_files)
        if shuffle:
            random.shuffle(image_files)

        train = PartiallyLabelledDataset()
        train.root = self.root
        train.image_files = image_files[num_val:]
        train.image_dir = self.image_dir
        train.label_dir = self.label_dir
        train.class_id2name = self.class_id2name
        train.class_name2id = self.class_name2id

        val = PartiallyLabelledDataset()
        val.root = self.root
        val.image_files = image_files[:num_val]
        val.image_dir = self.image_dir
        val.label_dir = self.label_dir
        val.class_id2name = self.class_id2name
        val.class_name2id = self.class_name2id

        return train, val
