from Labeller import LabelHelper, PartiallyLabelledDataset


if __name__ == '__main__':
    dataset = PartiallyLabelledDataset()
    dataset.load('datasets/sample')
    helper = LabelHelper(dataset)
    helper.mainloop()
