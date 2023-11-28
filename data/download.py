import datetime
from typing import Any

from pytube import Channel, YouTube
from configs import configs
from configs.log_conf import getLogger
from tqdm import tqdm
import os

LOGGER = getLogger(__name__)


class Download:
    def __init__(self, channel_url, dataset_name, num_videos):
        self.channel_url = channel_url
        self.dataset_name = dataset_name
        self.num_videos = num_videos

    @property
    def dataset_path(self):
        return os.path.join(configs.EXTRA_VIDEOS_PATH, self.dataset_name)

    @property
    def urls_file(self):
        return os.path.join(self.dataset_path, "urls.txt")

    def __call__(self) -> Any:
        self.run()

    def run(self):
        c = Channel(self.channel_url)
        if os.path.exists(self.urls_file):
            with open(self.urls_file, 'r') as f:
                existing_urls = [x.strip() for x in f.readlines()]
        else:
            os.makedirs(os.path.dirname(self.urls_file))
            existing_urls = []

        new_urls = existing_urls
        count = 0
        progress_bar = tqdm(total=self.num_videos)
        for video_url in c.video_urls:
            if video_url not in existing_urls:
                try: 
                    yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
                    stream = yt.streams.get_by_itag(itag=22)
                    stream.download(output_path=self.dataset_path, filename=f'yt_tennis_{str(len(existing_urls) + count + 1)}.mp4')
                    count += 1
                    progress_bar.update(1)
                    new_urls.append(video_url)
                except Exception as e:
                    continue 
                if count >= self.num_videos:
                    break
        with open(self.urls_file, 'w') as f:
            f.writelines([x + '\n' for x in new_urls])


