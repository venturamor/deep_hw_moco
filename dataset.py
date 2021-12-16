from torch.utils.data import Dataset
import pandas as pd


class OurDataSet(Dataset):
    def __init__(self, data_path, csv_path, transform):
        self.aug_transform = transform
        self.data_path = data_path
        self.csv_file = pd.read_csv(csv_path)


        print('done')



    # def __len__(self):

    # def __getitem__(self, item):


    # def apply_transform(self):


if __name__ == '__main__':
    csv_path = r"imagenette\noisy_imagenette.csv"
    dataset = OurDataSet('', csv_path, '')

