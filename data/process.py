"""Script for processing the videos into frames and flow frames"""

import os

from models.vision.flownet.run2s import generate_flows
from utils.video import video_to_frames
from configs.configs import VIDEOS_PATH, FRAMES_PATH, FLOWS_PATH


def vid2img(videos=('V006', 'V007', 'V008', 'V009', 'V010'), videos_dir=VIDEOS_PATH, frames_dir=FRAMES_PATH):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)    

    for video in videos:  # lets extract frames
        video_to_frames(video_path=os.path.join(videos_dir, video + '.mp4'),  # assuming .mp4
                        frames_dir=frames_dir,
                        chunk_size=1000)


def img2flw(frames_dir=FRAMES_PATH, flow_dir=FLOWS_PATH):
    generate_flows(image_dir=frames_dir, flow_dir=flow_dir)


def main(_argv):
    print("Video to Images")
    vid2img()

    print("Images to Flow")
    img2flw()


if __name__ == '__main__':
    main()
