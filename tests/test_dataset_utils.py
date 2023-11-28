import os
from configs.configs import FRAMES_PATH
from src.dataset_utils import CaptionUtils, DatasetUtils


class TestCaptionsUtils:
    def __init__(self):
        self.cutils = CaptionUtils(split_id='02')

    def test_convert_points_dict_to_list(self):
        _,_,_,point_dict = DatasetUtils().load_data()
        print(point_dict)

    def test_load_data_exists(self):
        print(f'caption data exists? {self.cutils.load_data_exists()}')

    def test_load_from_disk(self):
        samples, videos, events, points = self.cutils.load_from_disk()

    def test_load_data_when_captions(self):
        samples, videos, events, points = self.cutils.load_data_when_captions()


class TestDatasetUtils:
    def __init__(self):
        self.du = DatasetUtils(split_id='02')

    def test_check_classes(self):
        classes1 = self.du.classes_dict.keys()
        classes2 = self.du.get_classes()
        print(classes1)
        print(classes2)

    def test_num_classes(self):
        num = self.du.num_class
        print(f'num classes: {num}')

    def test_get_classes(self):
        classes = self.du.get_classes()
        print(f'classes: {classes}')

    def test_class_counts(self):
        c, counts, counts_dict = self.du.class_counts()
        print(f'c: {c}\ncounts: {counts}\ncounts_dict: {counts_dict}')

    def test_balance_classes(self):
        samples, _, _, _ = self.du.load_data()
        samples = self.du._balance_classes(samples)
        
    def test_load_data_exists(self):
        exists = self.du.load_data_exists()
        print(f'load data exists? {exists}')

    def test_get_captions(self):
        samples, _, _, points = self.du.load_from_disk()
        self.du.get_captions(samples, points)

    def test_generate_windows(self):
        samples, _, _, _ = self.du.load_from_disk()
        sample = samples[0]
        idx, _window, _every, _stride = self.du.generate_windows(_sample=sample, _window=8, _every=10, _stride= 4)
        print(f'idxs: {idx}\nwindow: {_window}\nevery {_every}\nstride: {_stride}')

    def test_load_data_when_captions(self):
        self.du._captions = True
        self.du.load_data_when_captions()

    def test_load_from_disk(self):
        samples, videos, events, points = self.du.load_from_disk()
        print(f'samples n0: {len(samples)}')
        print(f'videos: {videos}')
        print(f'events n0: {len(events)}')
        print(f'points n0: {len(points)}')

    def test_get_image_path(self):
        image_path = self.du.get_image_path(root_dir=FRAMES_PATH, video_name='V006', frame_number=49220, chunk_size=1000)
        print(f'image path: {image_path}')
        print(f'image path exists?: {os.path.exists(image_path)}')

    def test_load_data(self):
        self.du.load_data()

    def test_get_video_lengths(self):
        last = self.du.get_video_lengths()
        print(f'last: {last}')



if __name__ == '__main__':

    # ---------------------------------------------------------
    t = TestCaptionsUtils()
    # t.test_convert_points_dict_to_list()
    # t.test_load_data_exists()
    # t.test_load_from_disk()
    t.test_load_data_when_captions()

    # ---------------------------------------------------------
    # t = TestDatasetUtils()
    # t.test_check_classes()
    # t.test_num_classes()
    # t.test_get_classes()
    # t.test_class_counts()
    # t.test_load_data_exists()
    # t.test_get_captions()
    # t.test_generate_windows()
    # t.test_load_data_when_captions()
    # t.test_load_from_disk()
    # t.test_get_image_path()
    # t.test_load_data()
    # t.test_get_video_lengths()