from typing import Dict
import yt_dlp.utils
from yt_dlp import YoutubeDL
import os


class YouTubeTrackDownloader:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.tracks_dir = os.path.join(base_dir, 'tracks_wav')
        self.cache_dir = os.path.join(base_dir, 'tracks_cache')
        self.outtmpl = os.path.join(self.tracks_dir, '%(id)s.%(ext)s')

        self.common_ydl_opts = {
            'quiet': True,
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': self.outtmpl,
            'cachedir': self.cache_dir,
            'noplaylist': True,
            'extractaudio': True,
            'no_cache_dir': True,
        }

    def get_info(self, url) -> Dict[str, str]:
        with YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return {
            'video_id': info.get('id', ''),
            'title': info.get('title', ''),
            'uploader': info.get('uploader', ''),
            'description': info.get('description', ''),
            'duration': info.get('duration', 0),
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0),
            'comment_count': info.get('comment_count', 0),
            'categories': info.get('categories', []),
            'url': url
        }

    def download(self, url) -> str:
        with YoutubeDL(self.common_ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        return os.path.join('tracks_wav', f"{info.get('id')}.wav")
