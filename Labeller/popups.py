import tkinter as tk
from functools import partial


class MessageBox:
    def __init__(self, msg, cx, cy, title):
        self.root = tk.Tk()
        self.root.title(title)

        line_lengths = list(map(len, msg.splitlines()))
        longest_line_length = max(line_lengths)
        num_lines = len(line_lengths)
        # fit width to the longest line length
        width = 40 + 8 * longest_line_length
        # fit height to the number of lines
        height = 40 + 14 * num_lines
        self.root.geometry('{}x{}+{}+{}'.format(
            width, height,
            # popup at the center of the screen
            cx - width // 2, cy - height // 2
        ))

        self.label = tk.Label(self.root, text=msg)
        self.label.pack()
        tk.Button(self.root, text='Ok', command=self.close).pack()
        self.root.bind('<Return>', lambda x: self.close())
        self.root.bind('<Escape>', lambda x: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()


class MultipleChoiceQuestionAsker:
    def __init__(self, prompt: str, options: tuple,
                 cx, cy,
                 title='Options', width=300, height=150):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('{}x{}+{}+{}'.format(
            width, height, cx - width // 2, cy - height // 2
        ))
        self.value = 0

        tk.Label(self.root, text=prompt, wraplength=4 * width // 5).pack()
        for i, option in enumerate(options):
            button = tk.Radiobutton(
                self.root, text=option, value=i,
                command=partial(self.set_value, v=i)
            )
            if i == 0:
                button.select()
            button.pack(anchor="w")

        tk.Button(self.root, text="Ok", command=self.close).pack()
        tk.Button(self.root, text="Quit", command=self.quit).pack()
        self.root.bind('<Return>', lambda x: self.close())
        self.root.bind('<Escape>', lambda x: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def set_value(self, v):
        self.value = v

    def quit(self):
        self.set_value(-1)
        self.close()

    def close(self):
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
        return self.value


class YesNoQuestionAsker:
    def __init__(self, prompt: str, cx, cy, title='Options', width=200, height=100):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, cx - width // 2, cy - height // 2))
        self.value = None

        tk.Label(self.root, text=prompt, wraplength=4 * width // 5).pack()

        tk.Button(self.root, text="Yes", command=partial(self.set_value, v=True)).pack()
        tk.Button(self.root, text="No", command=partial(self.set_value, v=False)).pack()
        self.root.bind('<Return>', lambda x: self.set_value(True))
        self.root.bind('<Escape>', lambda x: self.set_value(False))
        self.root.protocol("WM_DELETE_WINDOW", partial(self.set_value, v=False))

    def set_value(self, v):
        self.value = v
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
        return self.value


class ScrollableMenubar():
    def __init__(self, l: list, width, height, x, y, digits=4):
        self.root = tk.Tk()
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        self.scrollbar = tk.Scrollbar(self.root)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(self.root, yscrollcommand=self.scrollbar.set)
        self.fill_listbox(l, digits)

        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.listbox.yview)

        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.root.grab_set()

    def mainloop(self):
        self.root.mainloop()

    def fill_listbox(self, l, digits):
        for idx, item in enumerate(l):
            self.listbox.insert(tk.END, '%0{}d: {}'.format(digits, item) % idx)

    def bind(self, on_select):
        if on_select is not None:
            self.listbox.bind('<<ListboxSelect>>', on_select)
        self.root.deiconify()

    def unbind(self):
        self.listbox.unbind('<<ListboxSelect>>')
        self.root.withdraw()

    def set_title(self, title):
        self.root.title(title)

    def close(self):
        self.root.quit()
        self.root.destroy()
