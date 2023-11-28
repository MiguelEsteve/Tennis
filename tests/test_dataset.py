
from configs.configs import DATASETS_PATH
from src.dataset import TennisSet


class TestTennisSet:
    def __init__(self):
        self.dataset = TennisSet(root=DATASETS_PATH)

    def test_get_classes(self):
        classes = self.dataset._get_classes()
        print(classes)

    def test_load_data(self):
        self.dataset.load_data()

    def test_get_video_lengths(self):
        vlengths = self.dataset._get_video_lengths()
        print(vlengths)

    def test__len__(self):
        l = self.dataset.__len__()
        print(l)

    def test_get_item(self):
        img,label, idx = self.dataset.__getitem__(0)
        print(f'img.shape: {img.shape}')
        print(f'label: {label}')
        print(f'idx: {idx}')


    def test_loader(self):
        from mxnet import gluon
        trainset = TennisSet(root=DATASETS_PATH,
                            captions=False,
                            transform=None,
                            split='train',
                            every=1,
                            window=1,
                            stride=1,
                            balance=True,
                            split_id='02',
                            flow=False,
                            max_cap_len=-1,
                            vocab=None,
                            inference=False,
                            feats_model=None,
                            save_feats=None,
                            model_id='0000')
        
        train_data = gluon.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)
        
        for i, batch in enumerate(train_data):
            print(i, batch)
            if i >= 1:
                break


if __name__ == '__main__':
    t = TestTennisSet()
    # t.test_get_classes()
    # t.test_load_data()
    # t.test_get_video_lengths()
    # t.test__len__()
    # t.test_get_item()

    t.test_loader()    