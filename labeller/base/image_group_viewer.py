import os
import tkinter as tk
from abc import *
from math import log10

from .image_window import ImageWindow
from .popups import ScrollableMenubar
from .utils import on_caps_lock_off


class ImageGroupViewer(ImageWindow, metaclass=ABCMeta):
    '''
<Group View Actions>
a or left arrow: go to the previous image
d or right arrow: go to the next image
Home: go to the first image
End: go to the last image
    '''

    def __init__(self, items: list, win_title=None, axes_pos=ImageWindow.DEFAULT_AXES_POSITION):
        super().__init__(win_title, axes_pos)
        self.items = items
        self.id = 0
        self.prev_id = None

        width = self.root.winfo_width() // 9
        height = self.root.winfo_height()
        x = self.root.winfo_x() + self.root.winfo_width()
        y = 0
        self.image_menubar = ScrollableMenubar(
            [os.path.basename(item) for item in self.items],
            width, height, x, y,
            int(log10(self.num_items)) + 1
        )
        self.image_menubar.set_title('Image List')
        self.enable_menubar()
        self.force_focus()

    def enable_menubar(self):
        self.image_menubar.bind(self.on_image_menubar_select)

    def disable_menubar(self):
        self.image_menubar.unbind()

    @property
    def num_items(self):
        return len(self.items)

    def should_update(self):
        if self.prev_id == self.id:
            return False
        else:
            self.ax.clear()
            if self.verbose:
                print('image id: {}'.format(self.id))
            self.prev_id = self.id
            return True

    def on_image_menubar_select(self, event):
        if self.callbacks_alive:
            # Note here that Tkinter passes an event object to onselect()
            selections = event.widget.curselection()
            if len(selections) > 0:
                self.id = int(selections[0])
                self.display()
                self.force_focus()

    def mainloop(self):
        super().mainloop()
        self.image_menubar.mainloop()

    def close(self):
        super().close()
        self.image_menubar.close()

    def remove_current_item(self):
        # clear listbox and remove current item
        self.image_menubar.listbox.delete(0, self.num_items - 1)
        del self.items[self.id]

        # reload listbox
        self.image_menubar.fill_listbox(
            [os.path.basename(item) for item in self.items],
            int(log10(self.num_items)) + 1
        )
        self.image_menubar.listbox.pack()

        self.id = self.id % self.num_items
        self.prev_id = (self.id - 1) % self.num_items

    @abstractmethod
    def display(self):
        try:
            self.image_menubar.listbox.selection_clear(0, self.num_items - 1)
            self.image_menubar.listbox.select_set(self.id, self.id)
        # tk.TclError is raised when either of selection_clear or select_set
        # is called when the image menubar is already closed
        except tk.TclError as e:
            pass
        if self.num_items == 0:
            raise Exception('No images to display')

    @on_caps_lock_off
    def on_key_press(self, event):
        super().on_key_press(event)
        if event.key in ['left', 'a']:
            self.id = (self.id - 1) % self.num_items
        elif event.key in ['right', 'd']:
            self.id = (self.id + 1) % self.num_items
        elif event.key == 'home':
            self.id = 0
        elif event.key == 'end':
            self.id = self.num_items - 1
