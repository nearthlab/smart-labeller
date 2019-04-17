import os
from skimage.io import imread
from .image_group_viewer import ImageGroupViewer



class ImageViewer(ImageGroupViewer):
    def __init__(self, directory):
        super(ImageViewer, self).__init__(
            sorted([os.path.join(directory, x)
                    for x in os.listdir(directory)
                    if x.lower().endswith('.jpg') or x.lower().endswith('.png') or x.lower().endswith('.bmp')
                    ]),
            os.path.basename(directory)
        )
        self.display()


    def display(self):
        super(ImageViewer, self).display()
        if self.should_update():
            image_file = self.items[self.id]
            img = imread(image_file)
            self.set_image(img)

            title = "{} ({}/{})".format(
                os.path.basename(image_file),
                self.id + 1,
                self.num_items
            )
            self.set_title(title)


    def on_key_press(self, event):
        super().on_key_press(event)
        self.display()
