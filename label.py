from Labeller import LabelHelper, PartiallyLabelledDataset


if __name__ == '__main__':
    dataset = PartiallyLabelledDataset()
    dataset.load('datasets/temp')
    helper = LabelHelper(dataset)
    helper.mainloop()
