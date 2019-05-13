import sys
import tkinter as tk
from tkinter import messagebox
from Labeller import PartiallyLabelledDataset, AugmentHelper
from Labeller.popups import ask_directory

if __name__ == '__main__':
    argc = len(sys.argv)
    dirname = ()
    if argc == 1:
        dirname = ask_directory()
    elif argc == 2:
        dirname = sys.argv[1]
    else:
        print('Usage: "{}" or "{} [/path/to/dataset]"'.format(sys.argv[0], sys.argv[0]))
        exit(1)

    try:
        if dirname is not None:
            # Load parially labelled dataset
            dataset = PartiallyLabelledDataset()
            dataset.load(dirname, labelled_only=True)

            # Create label helper
            helper = AugmentHelper(dataset)
            result = helper.mainloop()

            # Notify user
            if result is not None:
                messagebox.showinfo(
                    'Finished!', 'The augmented dataset is saved at {}'.format(result)
                )

    except Exception as e:
        messagebox.showerror('Failure', e)
