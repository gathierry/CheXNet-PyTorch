from sklearn.model_selection import train_test_split
import pandas as pd
import os


class CXR2:
    def __init__(self, data_dir, train_val):
        self.data_dir = data_dir
        this_path = os.path.abspath(os.path.dirname(__file__))
        fn_list_path = os.path.join(this_path, train_val + '.txt')
        self.image_files = []
        with open(fn_list_path, 'r') as ins:
            for i, line in enumerate(ins):
                l = line.rstrip()
                fn = l.split(' ')[0].split('/')[1]
                self.image_files.append(fn)
        self.entry_df = pd.read_csv(os.path.join(self.data_dir, 'Data_Entry_2017.csv'))
        labels = set()
        for label in self.entry_df['Finding Labels']:
            for l in label.split('|'):
                if l != 'No Finding':
                    labels.add(l)
        self.labels = sorted(list(labels))

    def size(self):
        return len(self.image_files)

    def get_image_path(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, 'images', image_file)
        return image_path

    def get_label(self, idx):
        image_file = self.image_files[idx]
        sub_df = self.entry_df[self.entry_df['Image Index'] == image_file]
        labels = sub_df.iloc[0]['Finding Labels']
        labels_set = set(labels.split('|'))
        label = [(l in labels_set)*1 for l in self.labels]
        return label

    def get_occurrences(self):
        sub_df = self.entry_df.loc[self.entry_df['Image Index'].isin(self.image_files)]
        occurrences = {l: 0 for l in self.labels}
        for label in sub_df['Finding Labels']:
            for l in label.split('|'):
                if l in occurrences:
                    occurrences[l] += 1
        return [occurrences[l] for l in self.labels]

    def get_nofinding_num(self):
        sub_df = self.entry_df.loc[self.entry_df['Image Index'].isin(self.image_files)]
        s = 0
        for label in sub_df['Finding Labels']:
            if label == 'No Finding':
                s += 1
        return s


if __name__ == '__main__':
    db_path = '/home/storage/NIH_Chest_Xray/'
    data_parser = CXR2(db_path, 'train')
    train = data_parser.get_occurrences()
    data_parser = CXR2(db_path, 'val')
    val = data_parser.get_occurrences()
    data_parser = CXR2(db_path, 'test')
    test = data_parser.get_occurrences()
    print(['%.3f'%(float(x)/z) for x, y, z in zip(train, val, test)])





