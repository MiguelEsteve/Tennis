import os

# Root folder for project code vs computer host
roots = {'DESKTOP-JV9DACD':
            {'Tennis': 'C:/repos/Tennis'},
        'PC-514445':
            {'Tennis': 'C:/repos/vct-ml-yolov8'}}
PROJECT_PATH = roots[os.getenv('computername')]['Tennis']

# Get Dataset paths
sources = {'DESKTOP-JV9DACD':
                {'datasets_path': 'G:/Datasets/TenniSet'},
            'PC-514445':
                {'datasets_path': 'I:/Datasets/TenniSet'}}

DATASETS_PATH = sources[os.getenv('computername')]['datasets_path']
MODEL_SPECIFIC_PATH = DATASETS_PATH + '/model_specific'

FRAMES_PATH = DATASETS_PATH + '/frames'
VIDEOS_PATH = DATASETS_PATH + '/videos'
FLOWS_PATH = DATASETS_PATH + '/flows'
SPLITS_PATH = DATASETS_PATH + '/splits'
ANNOTATIONS_PATH = DATASETS_PATH + '/annotations'
LABELS_PATH = ANNOTATIONS_PATH + '/labels'
LOAD_DATA_PATH = ANNOTATIONS_PATH + '/load_data'

EXTRA_VIDEOS_PATH = DATASETS_PATH + '/extra_videos'

WEIGHTS_FLOWNET_PATH = MODEL_SPECIFIC_PATH + '/weights/flownet'
WEIGHTS_MODEL_PATH = MODEL_SPECIFIC_PATH + '/weights/tenniset'
PREDICTS_PATH = MODEL_SPECIFIC_PATH + '/predicts'
SUMMARIES_PATH = MODEL_SPECIFIC_PATH + '/summaries'


PRETRAINED = DATASETS_PATH + '/pretrained'
TEST_IMAGES = DATASETS_PATH + '/test_images'
TEST_VIDEOS = DATASETS_PATH + '/test_videos'