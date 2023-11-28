import os
import csv
import json
import math
import numpy as np

from torch.utils.data import Dataset

from configs.configs import VIDEOS_PATH, FRAMES_PATH, SPLITS_PATH, LABELS_PATH, ANNOTATIONS_PATH, LOAD_DATA_PATH
from configs.log_conf import getLogger
from utils.video import video_to_frames
import gluonnlp

LOGGER = getLogger(__name__)

all_videos = os.listdir(VIDEOS_PATH)
all_frames = os.listdir(FRAMES_PATH)


class CaptionUtils:
    def __init__(self, split_id: str = '01', split:str = 'train', vocab=None, max_cap_len:int=-1):
        self._split_id = split_id
        self._split = split
        self._vocab = vocab
        self._max_cap_len = max_cap_len

        if not os.path.exists(self.samples_fn):
            os.makedirs(os.path.dirname(self.samples_fn), exist_ok=True)

        if not os.path.exists(self.points_fn):
            os.makedirs(os.path.dirname(self.points_fn), exist_ok=True)

    @property
    def samples_fn(self):
        return os.path.join(LOAD_DATA_PATH, self._split_id, 'captions', 'samples.csv')

    @property 
    def points_fn(self):
        return os.path.join(LOAD_DATA_PATH, self._split_id, 'captions', 'points.json')

    def load_data_exists(self):
        return os.path.exists(self.samples_fn) and os.path.exists(self.points_fn)

    def load_from_disk(self):
        samples = []
        with open(self.samples_fn, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                row[1] = int(row[1])
                samples.append(row)

        with open(self.points_fn, 'r') as file:
            points_dict = json.load(file)
            points_dict = self.convert_points_dict_to_np(points_dict)
        return samples, points_dict


    @staticmethod
    def convert_points_dict_to_np(points_dict:dict):
        converted = {}
        for k, v in points_dict.items():
            v[5] = np.array(v[5])
            converted[k] = v
        return converted


    @staticmethod
    def convert_points_dict_to_list(points_dict:dict):
        converted = {}
        for k, v in points_dict.items():
            v[5] = v[5].tolist()
            converted[k] = v
        return converted


    def load_data_when_captions(self, override:bool=False, save:bool=True):
        datasetutils = DatasetUtils(split_id=self._split_id, split=self._split)
        
        samples, videos,events, points_dict = datasetutils.load_data()
        
        if not override and self.load_data_exists():
            samples, points_dict = self.load_from_disk()
            return samples, videos, events, points_dict
        
        samples = list(points_dict.keys())
        caps = [p[4] for p in points_dict.values()]
        words = ''.join(caps).split()
        if self._vocab is None:
            counter = gluonnlp.data.count_tokens(words)
            self._vocab = gluonnlp.Vocab(counter)

            # TODO utilize when move all to pytorch !
            from collections import Counter
            counter_bis = Counter(words)
            vocab_bis = {word:index for index, word in enumerate(counter.keys(), 2)}
            vocab_bis['<BOS>'], vocab_bis['<EOS>'] = 0, 1
            
            for i in range(len(samples)):
                point_id = samples[i]
                cap = points_dict[point_id][4]
                if self._max_cap_len >= 0:
                    cap_ids = self._vocab[cap.split()[:self._max_cap_len]]
                else:
                    cap_ids = self._vocab[cap.split()]

                cap_ids.insert(0, self._vocab[self._vocab.bos_token])
                cap_ids.append(self._vocab[self._vocab.eos_token])
                # TODO change when move to pytorch
                cap_ids_bis = cap_ids
                cap_ids_bis.insert(0, vocab_bis['<BOS>'])
                cap_ids_bis.append(vocab_bis['<EOS>'])

                cap_ids = np.array(cap_ids, dtype=np.int32)
                points_dict[point_id].append(cap_ids)

        if save:

            with open(self.samples_fn, 'w', newline='') as file:
                writter = csv.writer(file)
                writter.writerows(samples)

            self.convert_points_dict_to_list(points_dict)
            with open(self.points_fn, 'w') as file:
                json.dump(points_dict, file)


        return samples, videos, events, points_dict


class DatasetUtils:
    def __init__(self, 
                    split: str = 'train', 
                    split_id:str='01'):
        self._split = split
        self._split_id = split_id

        if not os.path.exists(os.path.dirname(self.samples_fn)):
            os.makedirs(os.path.dirname(self.samples_fn), exist_ok=True)

        LOGGER.test('Dataset Util instantiated')
        
    @property
    def samples_fn(self):
        return os.path.join(LOAD_DATA_PATH, self._split_id, 'samples.csv')
    
    @property            
    def events_fn(self):
        return os.path.join(LOAD_DATA_PATH, self._split_id, 'events.csv')
                
    @property 
    def points_fn(self):
        return os.path.join(LOAD_DATA_PATH, self._split_id, 'points.json')
    
    @property
    def classes_fn(self):
        return os.path.join(ANNOTATIONS_PATH, 'classes.names')

    @property
    def classes_dict(self):
        return {'OTH': 'Other',
                'SFI': 'ServeFarIn', 'SFF': 'ServeFarFault', 'SFL': 'ServeFarLeft',
                'SNI': 'ServeNearIn', 'SNF': 'ServeNearFault', 'SNL':'ServeNearLeft',
                'HFL': 'HitFarLeft', 'HFR': 'HitFarRight', 'HNL': 'HitNearLeft', 'HNR': 'HitNearRight'}

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.get_classes())
    
    def get_classes(self):
        with open(self.classes_fn, 'r') as f:
            classes = [x.strip() for x in f.readlines()]
        return classes

    def class_counts(self, samples):
        classes = self.get_classes()
        counts = [0] * len(classes)
        counts_dict = {}
        for sample in samples:
            counts[classes.index(sample[2])] += 1
        
        for c, count in zip(classes, counts):
            counts_dict[c] = count
        
        return classes, counts, counts_dict

    def _balance_classes(self, samples):
        """
        Balance the dataset on 'Other' class, with next most sampled class, uses uniform random sampling

        Returns:
            list: the balanced set of samples
        """
        #
        counts = self.class_counts()
        next_most = max(counts[1:])
        ratio = next_most/float(counts[0]+1)

        balanced = list()
        for sample in samples:
            if sample[2] == 'OTH' and np.random.uniform(0, 1) > ratio:
                continue
            balanced.append(sample)
        samples = balanced

        return samples


    def load_data_exists(self):
        return os.path.exists(self.samples_fn) and os.path.exists(self.events_fn) and os.path.exists(self.points_fn)
    
    def generate_windows(self, _sample, _window=8, _every=4, _stride=1):

        _window_offsets = list(range(int(-_window/2), int(math.ceil(_window/2))))
        
        frame_idxs = []
        for offset in _window_offsets:
            max_frame_idx = self.get_video_lengths()[_sample[0] + '.mp4'] - _every
            for i in range(_every):
                u = (max_frame_idx - i)%_every
                if u == 0:
                    max_frame_idx -= i
                    break
            frame_idx = min(max(0, _sample[1]  + offset*_stride), max_frame_idx)
            frame_idxs.append(frame_idx)
        return frame_idxs, _window, _every, _stride


    def load_from_disk(self):
        samples = []
        with open(self.samples_fn, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                row[1] = int(row[1])
                samples.append(row)

        videos = set()
        for sample in samples:
            videos.add(sample[0])
        videos = list(videos)

        events = []
        with open(self.events_fn, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                row[1], row[2] = int(row[1]), int(row[2])
                events.append(row)

        with open(self.points_fn, 'r') as file:
            points_dict = json.load(file)

        return samples, videos, events, points_dict
        

    @staticmethod
    def get_image_path(root_dir:str, video_name:str, frame_number:int, chunk_size:int=1000):
        """
        Given a frame number, provides the frame path that contains the imafe
        Args:
            root_dir (str): the frames root dir
            video_name (str): the name of the video (with no extension .mp4)
            frame number (int): the index of the frame in the video
            chunk_size (int): the directory bloch size for frames   # TODO as global parameter
        """
        chunk = int(frame_number/chunk_size)*chunk_size
        return os.path.join(root_dir, video_name+'.mp4', '{:010d}'.format(chunk), '{:010d}.jpg'.format(frame_number))


    def load_data(self, split_id:str =None, override:bool=False, save:bool=True):
        """
        Load the data: samples, videos, events, points

        Args:
            split_id (str): the split id either '01' or '02'
            override (bool): if override existing data un disk
            save(bool): if save data to disk

        Returns:
            list: of samples [[video, frame, class], ...]
            list: of videos [video1, video2, ...]
            list: of events [[video, start_frame, last_frame, cur_class], ...]
            dict: of point (with trasncripts of actions)
        """

        split_id = self._split_id if split_id is None else split_id
        assert split_id == '01' or split_id == '02', f'wrong split_id {split_id}'

        if self.load_data_exists() and not override:
            return self.load_from_disk()

        splits_file = os.path.join(SPLITS_PATH, split_id, self._split + '.txt')

        # load the splits file
        if os.path.exists(splits_file):
            LOGGER.info("Loading data from {}".format(splits_file))
            with open(os.path.join(SPLITS_PATH, split_id, self._split + '.txt'), 'r') as f:
                lines = f.readlines()
                samples = [[line.rstrip().split()[0], int(line.rstrip().split()[1])] for line in lines]

            # make a list of the videos
            videos = set()
            for s in samples:
                videos.add(s[0])
            videos = list(videos)

            labels = dict()
            for video in videos:
                labels[video] = dict()

            # verify images exist, if not try and extract, if not again then ignore
            for i in range(2):  # go around twice, so if not all samples found extract, then re-check
                samples_exist = list()
                samples_exist_flag = True

                for s in samples:
                    if not os.path.exists(self.get_image_path(FRAMES_PATH, s[0], s[1])):
                        if i == 0:  # first attempt checking all samples exist, try extracting
                            samples_exist_flag = False  # will flag to extract frames

                            LOGGER.info("{} does not exist, will extract frames."
                                        "".format(self.get_image_path(self._frames_dir, s[0], s[1])))
                            break

                        else:  # second attempt, just ignore samples
                            LOGGER.info("{} does not exist, will ignore sample."
                                        "".format(self.get_image_path(self._frames_dir, s[0], s[1])))
                    else:
                        samples_exist.append(s)

                if samples_exist_flag:  # all samples exist
                    break
                else:
                    for video in videos:  # lets extract frames
                        video_to_frames(video_path=os.path.join(VIDEOS_PATH, video + '.mp4'),  # assuming .mp4
                                        frames_dir=self._frames_dir,
                                        chunk_size=1000)

            samples = samples_exist

            # load the class labels for each sample
            for video in videos:
                with open(os.path.join(LABELS_PATH, video + '.txt'), 'r') as f:
                    lines = f.readlines()
                    lines = [l.rstrip().split() for l in lines]

                for line in lines:
                    labels[video][int(line[0])] = line[1]

            # a dict of the frames in the set for each video
            in_set = dict()
            for video in videos:
                in_set[video] = list()

            #  add class labels to each sample
            for i in range(len(samples)):
                samples[i].append(labels[samples[i][0]][samples[i][1]])
                in_set[samples[i][0]].append(samples[i][1])

            # load events (consecutive frames with same class label)
            events = list()
            for video in in_set.keys():
                cur_class = 'OTH'
                start_frame = -1
                for frame in sorted(in_set[video]):
                    if start_frame < 0:
                        start_frame = frame
                        last_frame = frame
                    if labels[video][frame] != cur_class:
                        events.append([video, start_frame, last_frame, cur_class])
                        cur_class = labels[video][frame]
                        start_frame = frame 
                    last_frame = frame
                events.append([video, start_frame, last_frame, cur_class])  # add the last event

            # let's make up the points data
            with open(os.path.join(ANNOTATIONS_PATH, 'points.txt'), 'r') as f:
                lines = f.readlines()
            points = [l.rstrip().split() for l in lines]


            # add caps
            with open(os.path.join(ANNOTATIONS_PATH, 'captions.txt'), 'r') as f:
                lines = f.readlines()
                lines = [l.rstrip().split('\t') for l in lines]
            caps = dict()
            for l in lines:
                caps[l[0]] = l[1]
            for i in range(len(points)):
                points[i].append(caps[points[i][0]])

            # filter out points not in this split
            points_dict = dict()
            for point in points:
                if point[1] in videos and int(point[2]) in in_set[point[1]]:
                    points_dict[point[0]] = point[1:]
        
            if save:

                with open(self.samples_fn, 'w', newline='') as file:
                    writter = csv.writer(file)
                    writter.writerows(samples)

                with open(self.events_fn, 'w', newline='') as file:
                    writter = csv.writer(file)
                    writter.writerows(events)

                with open(self.points_fn, 'w', newline='') as file:
                    json.dump(points_dict, file)

            return samples, videos, events, points_dict
        else:
            LOGGER.info("Split {} does not exist, please make sure it exists to load a dataset.".format(splits_file))
            return None, None, None


    @staticmethod
    def get_video_lengths():
        lengths = {}
        for video_name in all_videos:
            largest_dir = sorted(os.listdir(os.path.join(FRAMES_PATH, video_name)))[-1]
            largest_file = sorted(os.listdir(os.path.join(FRAMES_PATH, video_name, largest_dir)))[-1]
            lengths[video_name] = int(largest_file[:-4])

        return lengths
    

class TennisSet(Dataset):
    def __init__(self, split_id:str='01', split:str = 'train', captions: bool = False) -> None:
        self._split_id = split_id
        self._split = split
        self.dataset_utils = DatasetUtils(split_id=split_id, split=split) if captions is False else CaptionUtils(split_id=split_id, split=split)

