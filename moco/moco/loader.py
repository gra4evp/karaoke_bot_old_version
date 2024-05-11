# -*- coding: utf-8 -*-
import threading
import os
import random
import librosa
import numpy as np
from audio_processing import get_fourier_chunks, trim_audio


class ThreadSafeIter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, iterator):
        self.it = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


class RandomIndexGenerator:  # Выдает случайные индексы, при достижении последнего индекса, начинает перебор заново
    def __init__(self, max_index):
        self.max_index = max_index
        self.indices = list(range(max_index))
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index == 0:
            random.shuffle(self.indices)  # Перемешиваем индексы перед каждым новым циклом

        if self.current_index >= self.max_index:
            self.current_index = 0  # Начинаем заново, если достигли конца списка индексов

        result = self.indices[self.current_index]
        self.current_index += 1
        return result


class SpectrogramDataset:

    def __init__(
            self,
            datapath: str,
            batch_size: int,
            sr: float,
            n_fft: int,
            hop_length: int,
            n_chunks: int,
            duration: float,
            max_chunk_extractions: int = 32
    ):
        self.datapath = datapath
        
        self.data = []
        for filename in os.listdir(datapath):
            filepath = os.path.join(datapath, filename)

            if os.path.getsize(filepath) <= 130000 * 1024:
                self.data.append(filename)

        if '.ipynb_checkpoints' in self.data:
            self.data.remove('.ipynb_checkpoints')

        self.batch_size = batch_size
        self.batch = []

        self.objects_idx_generator = ThreadSafeIter(RandomIndexGenerator(len(self.data)))

        # Частотные атрибуты для получения спектрограмм
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chunks = n_chunks
        # Временные атрибуты для получения спектрограмм
        self.duration = duration

        self.lock = threading.Lock()
        self.batch_lock = threading.Lock()

        self.max_chunk_extractions = max_chunk_extractions

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Возвращает спектрограмму := ||Амплитуды фурье образа|| из одного трека датасета и длительность трека
        '''
        filename = self.data[idx]
        
        audio_tensor, sr = librosa.load(path=os.path.join(self.datapath, filename), sr=self.sr, mono=True)
        audio_tensor = trim_audio(audio_tensor, self.sr, start_time=10, end_time=int(len(audio_tensor/sr) - 10))
        track_length = len(audio_tensor) // self.sr
        
        f_img = librosa.stft(audio_tensor, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = np.abs(np.real(f_img))

        return spectrogram, track_length, filename
    
    def __iter__(self):
        # Возращаем итератор раз используем yield
        while True:
            with self.lock:
                obj_idx = next(self.objects_idx_generator)

            try:
                spectrogram, track_length, filename = self[obj_idx]  # Получили спектрограмму всего трека
            except Exception as e:
                print(f"ВОЗНИКЛА ОШИБКА {e}")
            else:
                chunks_added = 0
                local_batch_id = None
                while chunks_added < self.max_chunk_extractions:

                    with self.batch_lock:
                        chunk_has_been_added = id(self.batch) == local_batch_id
                        
                        if not chunk_has_been_added:  # Если поток ещё не добавлял свой chunk
                            if len(self.batch) < self.batch_size:  # Если есть куда добавлять
                                chunks = self._get_spectrogram_chunks(spectrogram, track_length)  # Вырезаем куски
                                self.batch.append(chunks)
                                chunks_added += 1

                                local_batch_id = id(self.batch)

                            if len(self.batch) == self.batch_size:
                                batch = np.array(self.batch)
                                self.batch = []
                                yield batch
                        
    def _get_spectrogram_chunks(self, spectrogram: np.ndarray, track_length: int) -> np.ndarray:
        # Подбираем случайную длительность по пересечению при аугментациях 
        overlap = np.random.randint(self.duration // 2, self.duration)
        
        start_min = 5
        start_max = track_length - self.duration - (self.n_chunks - 1) * (self.duration - overlap) - 10
        start_time = np.random.randint(start_min, int(start_max))

        # Получили куски спектрограмм, т.к и спектрограмма и фурье образ одной размерности, а вырез по индексам
        chunks = get_fourier_chunks(
            f_img=spectrogram,
            sr=self.sr,
            n_chunks=self.n_chunks,
            hop_length=self.hop_length,
            start_time=start_time,
            duration=self.duration,
            overlap=overlap
        )
        
        return chunks


class SpectrogramDatasetInference:

    def __init__(
            self,
            datapath: str,
            batch_size: int,
            sr: float,
            n_fft: int,
            hop_length: int,
            n_chunks: int,
            duration: float,
            overlap: float,
            max_chunk_extractions: int = 32
    ):
        self.datapath = datapath
        
        self.data = []
        for filename in os.listdir(datapath):
            filepath = os.path.join(datapath, filename)

            if os.path.getsize(filepath) <= 130000 * 1024:
                self.data.append(filename)

        if '.ipynb_checkpoints' in self.data:
            self.data.remove('.ipynb_checkpoints')

        self.batch_size = batch_size
        self.batch = []

        self.objects_idx_generator = ThreadSafeIter(iter(range(len(self.data))))

        # Частотные атрибуты для получения спектрограмм
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chunks = n_chunks
        # Временные атрибуты для получения спектрограмм
        self.duration = duration
        self.overlap = overlap

        self.lock = threading.Lock()
        self.batch_lock = threading.Lock()

        self.max_chunk_extractions = max_chunk_extractions

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Возвращает спектрограмму := ||Амплитуды фурье образа|| из одного трека датасета и длительность трека
        '''
        filename = self.data[idx]
        
        audio_tensor, sr = librosa.load(path=os.path.join(self.datapath, filename), sr=self.sr, mono=True)
        audio_tensor = trim_audio(audio_tensor, self.sr, start_time=10, end_time=int(len(audio_tensor/sr) - 10))
        track_length = len(audio_tensor) // self.sr
        
        f_img = librosa.stft(audio_tensor, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = np.abs(np.real(f_img))

        return spectrogram, track_length, filename
    
    def __iter__(self):
        # Возращаем итератор раз используем yield
        while True:
            with self.lock:
                obj_idx = next(self.objects_idx_generator)

            try:
                spectrogram, track_length, filename = self[obj_idx]  # Получили спектрограмму всего трека
            except Exception as e:
                print(f"ВОЗНИКЛА ОШИБКА {e}")
            else:
                chunks_added = 0
                while chunks_added < self.max_chunk_extractions:
                    # print(f'Я поток который обрабатывает файл {filename} уже в {chunks_added + 1}')

                    with self.batch_lock:
                        if len(self.batch) < self.batch_size:  # Если есть куда добавлять
                            chunks = self._get_spectrogram_chunks(spectrogram, track_length)  # Вырезаем куски
                            self.batch.append((filename, np.array(chunks)))
                            chunks_added += 1

                        if len(self.batch) == self.batch_size:
                            batch = self.batch
                            self.batch = []
                            yield batch
                        
    def _get_spectrogram_chunks(self, spectrogram: np.ndarray, track_length: int) -> np.ndarray:
        start_min = 5
        start_max = track_length - self.duration - (self.n_chunks - 1) * (self.duration - self.overlap) - 10
        start_time = np.random.randint(start_min, int(start_max))

        # Получили куски спектрограмм, т.к и спектрограмма и фурье образ одной размерности, а вырез по индексам
        chunks = get_fourier_chunks(
            f_img=spectrogram,
            sr=self.sr,
            n_chunks=self.n_chunks,
            hop_length=self.hop_length,
            start_time=start_time,
            duration=self.duration,
            overlap=self.overlap
        )
        
        return chunks


class SpectrogramDatasetInferenceSingle:

    def __init__(
            self,
            datapath: str,
            sr: float,
            n_fft: int,
            hop_length: int,
            n_chunks: int,
            duration: float,
            overlap: float,
    ):
        self.datapath = datapath
        
        self.data = []
        for filename in os.listdir(datapath):
            filepath = os.path.join(datapath, filename)

            if os.path.getsize(filepath) <= 130000 * 1024:
                self.data.append(filename)

        if '.ipynb_checkpoints' in self.data:
            self.data.remove('.ipynb_checkpoints')

        # Частотные атрибуты для получения спектрограмм
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chunks = n_chunks
        # Временные атрибуты для получения спектрограмм
        self.duration = duration
        self.overlap = overlap

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Возвращает название трека, и кусок спектрограммы := ||Амплитуды фурье образа||
        '''
        try:
            filename = self.data[idx]

            audio_tensor, sr = librosa.load(path=os.path.join(self.datapath, filename), sr=self.sr, mono=True)
            audio_tensor = trim_audio(audio_tensor, self.sr, start_time=10, end_time=int(len(audio_tensor/sr) - 10))
            track_length = len(audio_tensor) // self.sr

            f_img = librosa.stft(audio_tensor, n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram = np.abs(np.real(f_img))
        
        except Exception as e:
            print(f"ВОЗНИКЛА ОШИБКА {e}")
        else:
            chunks = self._get_spectrogram_chunks(spectrogram, track_length)  # Вырезаем куски
        
        return filename, np.array(chunks)
                        
    def _get_spectrogram_chunks(self, spectrogram: np.ndarray, track_length: int) -> np.ndarray:
        start_min = 5
        start_max = track_length - self.duration - (self.n_chunks - 1) * (self.duration - self.overlap) - 10
        start_time = np.random.randint(start_min, int(start_max))

        # Получили куски спектрограмм, т.к и спектрограмма и фурье образ одной размерности, а вырез по индексам
        chunks = get_fourier_chunks(
            f_img=spectrogram,
            sr=self.sr,
            n_chunks=self.n_chunks,
            hop_length=self.hop_length,
            start_time=start_time,
            duration=self.duration,
            overlap=self.overlap
        )
        
        return chunks
