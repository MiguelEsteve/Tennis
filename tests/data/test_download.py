from data.download import Download


class TestDownload:
    def __init__(self):
        pass

    def test_pytube_description(self):
        
        from pytube import Channel, YouTube
        # c = Channel(url='https://www.youtube.com/c/movieclips')
        c = Channel(url='https://www.youtube.com/c/usopen')
        # c = Channel(url='https://www.youtube.com/c/rolandgarros')
        print(len(c))
        input('S')
        for i, video_url in enumerate(c.video_urls):
            print(f'{i}, {video_url}')
            yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
            try:
                print(yt.streams)
                stream = yt.streams.get_by_itag(itag=22)
                print(stream)
                print(yt.description)
                print(f'{i}, {f}')
                if i >= 10:
                    break
            except Exception as e:
                print(e)
                continue

    def test_run(self):
        m = Download(channel_url='https://www.youtube.com/c/usopen', 
                    dataset_name='usopen_test_1',
                    num_videos=8)
        m.run()


if __name__ == "__main__":
    t = TestDownload()
    # t.test_pytube_description()
    t.test_run()
