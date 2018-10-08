from __future__ import unicode_literals
import youtube_dl

ydl_urls = ['https://www.youtube.com/watch?v=GihybX7JyG4']


for i in range(len(ydl_urls)):

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }] ,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([ydl_urls[i]])


