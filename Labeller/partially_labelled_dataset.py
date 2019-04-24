import json
import os
import numpy as np

from skimage.io import imread
from .geometry import polygons_to_mask, mask_to_polygons, Polygon, extract_bbox_multi


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

    def __str__(self):
        return json.dumps(self.json())

    @property
    def bbox(self):
        return extract_bbox_multi(self.polys)

    def mask(self, shape):
        return polygons_to_mask(self.polys, shape)

    def is_empty(self):
        return len(self.polys) == 0 or self.class_id == -1


# annos: list of ObjectAnnotation objects
def dump_annotations(annos, file_path):
    with open(file_path, 'w') as fp:
        json.dump([a.json() for a in annos], fp)


def load_annotations(file_path):
    with open(file_path, 'r') as fp:
        raw_annos = json.load(fp)

    annos = []
    for raw_anno in raw_annos:
        annos.append(ObjectAnnotation(json.dumps(raw_anno)))
    return annos


class PartiallyLabelledDataset:
    def __init__(self):
        self.root = None
        self.image_files = []
        self.class_id2name = []
        self.class_name2id = []
        self.num_classes = 0

    @property
    def num_images(self):
        return len(self.image_files)

    def __len__(self):
        return self.num_images

    def load(self, dataset_dir, filter_unlabelled=False):
        assert os.path.isdir(dataset_dir), 'No such directory: {}'.format(dataset_dir)
        self.root = dataset_dir

        self.image_dir = os.path.join(self.root, 'images')
        self.label_dir = os.path.join(self.root, 'objects')

        assert os.path.isdir(self.image_dir), 'The dataset root directory {} must contain a subdirectory names images'.format(self.image_dir)
        verify_or_create_directory(self.label_dir)

        self.image_files = [os.path.join(self.image_dir, filename) \
                            for filename in sorted(os.listdir(self.image_dir)) \
                            if filename.lower()[-4:] in ['.jpg', '.png', '.bmp']]
        if filter_unlabelled:
            self.image_files = [path for image_id, path in enumerate(self.image_files) if not os.path.isfile(self.infer_label_path(image_id))]
        assert self.num_images > 0, 'No image found in the image directory: {}'.format(self.image_dir)

        class_label_path = os.path.join(self.root, 'class_names.json')
        assert os.path.isfile(class_label_path), 'Could not find class_names.json file in the directory: {}'.format(self.root)
        with open(class_label_path, 'r') as fp:
            self.class_id2name = json.load(fp)
            self.class_name2id = {name: id for id, name in enumerate(self.class_id2name)}
            self.num_classes = len(self.class_id2name)

    def infer_label_path(self, id):
        return os.path.join(self.label_dir, os.path.basename(self.image_files[id])[:-4] + '.json')

    def load_image(self, id):
        return imread(self.image_files[id])

    def load_annotations(self, id):
        label_path = self.infer_label_path(id)
        if os.path.isfile(label_path):
            return load_annotations(label_path)
        else:
            return []


def verify_or_create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(path):
        raise Exception('Could not find or create directory {}'.format(path))
