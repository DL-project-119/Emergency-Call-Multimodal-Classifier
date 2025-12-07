import os
import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from joblib import Parallel, delayed

SR = 16000
N_MFCC = 40
TARGET_T = 300

# 폴더에서 wav 인덱싱
def build_wav_index(audio_root, folders):
    file_map = {}

    for folder in folders:
        path = os.path.join(audio_root, folder)
        if not os.path.exists(path):
            continue

        for fp in glob.glob(os.path.join(path, "**/*.wav"), recursive=True):
            fid = os.path.splitext(os.path.basename(fp))[0]
            file_map[fid] = fp

    print(f"총 {len(file_map)}개 wav 파일 인덱싱 완료")
    return file_map


# MFCC 추출 + 정규화 + 보간
def extract_mfcc(filepath):
    try:
        y, _ = librosa.load(filepath, sr=SR)
        if y is None or len(y) == 0:
            return np.zeros((TARGET_T, N_MFCC), dtype=np.float32)

        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)

        # 정규화
        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True) + 1e-6
        mfcc = (mfcc - mean) / std

        T_orig = mfcc.shape[1]
        old_t = np.linspace(0, 1, T_orig)
        new_t = np.linspace(0, 1, TARGET_T)

        interp = np.zeros((N_MFCC, TARGET_T), dtype=np.float32)
        for i in range(N_MFCC):
            interp[i] = np.interp(new_t, old_t, mfcc[i])

        return interp.T  # (300, 40)

    except Exception:
        return np.zeros((TARGET_T, N_MFCC), dtype=np.float32)


#  CSV → MFCC 시퀀스 생성 → NPY 저장
def build_audio_sequence(csv_path, file_map, save_path):
    df = pd.read_csv(csv_path)
    ids = df["ID"].astype(str).values

    print(f"\n{csv_path} → MFCC 변환 시작 (총 {len(ids)}개)")

    features = Parallel(n_jobs=4)(
        delayed(lambda fid: extract_mfcc(file_map.get(fid, None)))(fid)
        for fid in tqdm(ids)
    )

    X = np.stack(features, axis=0)  # (N, 300, 40)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, X)

    print(f"저장 완료: {save_path}, shape={X.shape}")
    return X


if __name__ == "__main__":

    AUDIO_ROOT = "./data/Training/Audio/음성data"

    target_folders = [
        "질병(중증)", "질병(중증 외)", "자살", "임산부", "일반화재", "약물중독",
        "안전사고", "심정지", "산불", "사고", "부상", "대물사고",
        "기타화재", "기타구조", "기타구급", "기타"
    ]

    file_map = build_wav_index(AUDIO_ROOT, target_folders)

    build_audio_sequence(
        csv_path="./data/Training/Audio/ver1/X_audio_train_cleaned.csv",
        file_map=file_map,
        save_path="./data/Training/Audio/X_audio_train_seq.npy"
    )
