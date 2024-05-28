from typing import Dict
import yt_dlp.utils
from yt_dlp import YoutubeDL
import os


class YouTubeTrackDownloader:

    def __init__(self, url):
        self.url = url
        self.info: Dict[str, str] = {}
        self.filename: str = ''
        self.abspath = ''
        self.genres = None
        self.features = None

    def get_info(self) -> Dict[str, str]:
        if not self.info:
            ydl_opts = {'quiet': True}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                self.info = {
                    'video_id': info.get('id', ''),
                    'title': info.get('title', ''),
                    'uploader': info.get('uploader', ''),
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'comment_count': info.get('comment_count', 0),
                    'categories': info.get('categories', [])
                }
        return self.info

    def download(self) -> str:
        outtmpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tracks_wav', '%(id)s.%(ext)s')
        cachedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tracks_cache')
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': outtmpl,
            'cachedir': cachedir,
            'noplaylist': True,
            'extractaudio': True,
            'no_cache_dir': True,
            'quiet': True
        }
        with YoutubeDL(ydl_opts) as ydl:
            # error_code = ydl.download(url)
            info = ydl.extract_info(self.url, download=True)

        self.filename = 'tracks_wav/' + info.get('id') + '.wav'
        return self.filename

    def __str__(self) -> str:
        return f"YouTubeTrack: title - {self.info['title']} id - {self.info['video_id']}"

    def __repr__(self) -> str:
        return f"YouTubeTrack(url='{self.url}')"
