import sys
import traceback
from datetime import datetime

import matplotlib.pyplot as plt

from labeller import (
    TagHelper, ask_file, MessageBox
)

plt.rcParams['font.family'] = 'BM HANNA 11yrs old'
plt.rcParams['font.size'] = 15

if __name__ == '__main__':
    argc = len(sys.argv)
    filename = None
    if argc == 1:
        filename = ask_file(
            'Select a *.json file.',
            filetypes={'JSON file': '.json'}
        )
    elif argc == 2:
        filename = sys.argv[1]
    else:
        print('Usage: "{}" or "{} [/path/to/dataset]"'.format(sys.argv[0], sys.argv[0]))
        exit(1)

    try:
        if filename is not None:
            # Create tag helper
            helper = TagHelper(filename)
            helper.mainloop()

    except Exception as e:
        with open('{}.error.log'.format(sys.argv[0]), 'a') as fp:
            fp.write('{}\n'.format(datetime.now()))
            fp.write('-' * 40 + '\n')
            traceback.print_exc(file=fp)
            fp.write('-' * 40 + '\n')
        win = MessageBox(e, 'Error')
        win.mainloop()
