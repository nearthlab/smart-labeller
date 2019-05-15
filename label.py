import os
import sys
import json

from Labeller import LabelHelper, PartiallyLabelledDataset
from Labeller.popups import ask_directory, MessageBox

if __name__ == '__main__':
    argc = len(sys.argv)
    dirname = None
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
            dataset.load(dirname)

            # Load info.json if it is found under the dataset root directory
            info_path = os.path.join(dirname, 'info.json')
            if os.path.isfile(info_path):
                with open(info_path, 'r') as fp:
                    info = json.load(fp)
            else:
                info = None

            # Create label helper
            helper = LabelHelper(dataset, info)
            helper.mainloop()

    except Exception as e:
        win = MessageBox(str(e), 'Failure')
        win.mainloop()
