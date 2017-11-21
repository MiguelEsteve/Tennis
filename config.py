"""
Handles the file paths and model parameters

"""

from easydict import EasyDict as edict
import os

config = edict()


# Paths of directories that store the data
config.directories = edict()

# config.directories.main = '~/Tennis/' TODO: put this line back before final release
config.directories.main = '/media/hayden/Storage21/Tennis/'
config.directories.videos = os.path.join(config.directories.main, 'data', 'videos')
config.directories.annotations = os.path.join(config.directories.main, 'data', 'annotations')


# Annotator defaults
config.annotator = edict()

config.annotator.video_file = 'V007.mp4'  # must appear in config.directories.videos directory
config.annotator.classes_file = 'classes.txt'  # must appear in config.directories.annotations directory
