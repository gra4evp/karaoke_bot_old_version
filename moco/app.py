from flask import Flask, request, jsonify
import threading
import queue
import time
from downloader.downloader import YouTubeTrackDownloader
import os


app = Flask(__name__)

tasks = {}
task_counter = 1
download_queue = queue.Queue()


def downloading():
    downloader = YouTubeTrackDownloader()
    while True:
        task_id, url = download_queue.get()
        tasks[task_id]['status'] = 'downloading'
        info = downloader.get_info(url=url)
        # Обработку с Json
        filename = downloader.download(url=url)



@app.route('/submit', methods=['POST'])
def submit_task():
    global task_counter
    task_id = task_counter
    task_counter += 1
    tasks[task_id] = {'status': 'submitted'}
    # Здесь можно добавить логику для отправки задания в модель
    return jsonify({'task_id': task_id})


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
