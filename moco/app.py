import os
import time
import threading
import queue
from flask import Flask, request, jsonify
from downloader.downloader import YouTubeTrackDownloader

app = Flask(__name__)

tasks = {}
task_counter = 1
download_queue = queue.Queue()


def downloading():
    downloader = YouTubeTrackDownloader()
    while True:
        task_id, url = download_queue.get(block=True)
        tasks[task_id]['status'] = 'downloading'

        try:
            info = downloader.get_info(url=url)
        except Exception as e:
            print(f'ВОЗНИКЛА ОШИБКА ПРИ ПОЛУЧЕНИИ ИНФОРМАЦИИ ПО ССЫЛКЕ: {url}')
            print(e)
        else:
            video_id = info.get('video_id')
            if video_id is not None:
                filename = f'{video_id}.wav'
                if filename in os.listdir(downloader.tracks_dir):
                    filename = downloader.download(url=url)
                    tasks[task_id]['status'] = 'downloaded'
            else:
                tasks[task_id]['status'] = 'downloading_failed'


@app.route('/submit', methods=['POST'])
def submit_task():
    data = request.json
    global task_counter
    task_id = task_counter
    task_counter += 1
    tasks[task_id] = {'status': 'submitted'}.update(data)
    download_queue.put(tasks[task_id], block=True)
    return jsonify(tasks[task_id])


@app.route('/status/<int:task_id>', methods=['GET'])
def get_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify({'status': task['status']})


@app.route('/result/<int:task_id>', methods=['GET'])
def get_result(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    # Здесь можно добавить логику для получения результата из модели
    return jsonify({'result': 'Result for task {}'.format(task_id)})


if __name__ == '__main__':
    downloader_thread = threading.Thread(target=downloading, daemon=True)
    downloader_thread.start()
    app.run(host='0.0.0.0', port=5000)
