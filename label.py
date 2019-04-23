import sys
from Labeller import LabelHelper, PartiallyLabelledDataset


if __name__ == '__main__':
    dataset = PartiallyLabelledDataset()
    dataset.load(sys.argv[1])
    helper = LabelHelper(dataset)
    helper.mainloop()
