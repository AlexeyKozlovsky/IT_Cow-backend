import math
import json
import vosk
import librosa

import numpy as np
import pandas as pd

from pydub import AudioSegment


class SpeechRecognizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = vosk.Model(self.model_path)

    def fit(self, model_path):
        self.model_path = model_path

    def convert_to_text(self, audio_path):
        # TODO: дописать конвертацию в формат wav
        vosk.SetLogLevel(-1)
        sample_rate = 16000
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        int16 = np.int16(audio * 32768).tobytes()

        model = self.model
        recognizer = vosk.KaldiRecognizer(model, sample_rate)

        res = self._transcribe_words(recognizer, int16)

        new_list = sorted(res, key=lambda k: k['start'])
        return ' '.join([value['word'] for value in new_list])

    def _extract_words(self, res):
        jres = json.loads(res)
        if not 'result' in jres:
            return []
        words = jres['result']
        return words

    def _transcribe_words(self, recognizer, bytes):
        result = []

        chunk_size = 4000
        for chunk_no in range(math.ceil(len(bytes) / chunk_size)):
            start = chunk_no * chunk_size
            end = min(len(bytes), (chunk_no + 1) * chunk_size)
            data = bytes[start: end]

            if recognizer.AcceptWaveform(data):
                words = self._extract_words(recognizer.Result())
                result += words
        result += self._extract_words(recognizer.FinalResult())

        return result
