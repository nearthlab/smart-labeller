import os

from ..base import (
    ImageGroupViewer, on_caps_lock_off, load_rgb_image
)


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
            img = load_rgb_image(image_file)
            self.set_image(img)

            title = "{} ({}/{})".format(
                os.path.basename(image_file),
                self.id + 1,
                self.num_items
            )
            self.set_title(title)

    @on_caps_lock_off
    def on_key_press(self, event):
        super().on_key_press(event)
        self.display()
