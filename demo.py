import os
import json
import torch
import librosa
import numpy as np
import whisper
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SR = 16000
N_MFCC = 40
TARGET_T = 300

genai.configure(api_key="API_KEY")  
GEMINI_MODEL = "models/gemini-2.5-flash"
location_model = genai.GenerativeModel(GEMINI_MODEL)
summary_model = genai.GenerativeModel(GEMINI_MODEL)

# Whisper STT
whisper_model = whisper.load_model("large-v3")

def run_stt(wav_path):
    result = whisper_model.transcribe(wav_path, fp16=True)
    return result["text"].strip()


# ELECTRA tokenizer (텍스트 입력 처리)
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")


# MFCC 특징 추출
def extract_mfcc_sequence(wav_path, sr=SR, n_mfcc=N_MFCC, target_t=TARGET_T):
    try:
        y, _ = librosa.load(wav_path, sr=sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        mean = mfcc.mean(axis=1, keepdims=True)
        std  = mfcc.std(axis=1, keepdims=True) + 1e-6
        mfcc = (mfcc - mean) / std

        T_orig = mfcc.shape[1]
        if T_orig < 2:
            return np.zeros((target_t, n_mfcc), dtype=np.float32)

        old_x = np.linspace(0, 1, T_orig)
        new_x = np.linspace(0, 1, target_t)

        resampled = np.zeros((n_mfcc, target_t), dtype=np.float32)
        for i in range(n_mfcc):
            resampled[i] = np.interp(new_x, old_x, mfcc[i])

        return resampled.T.astype(np.float32)

    except Exception as e:
        print("[ERROR] Feature extraction failed:", e)
        return np.zeros((target_t, n_mfcc), dtype=np.float32)


# 텍스트 + 오디오 입력 생성
def make_inputs(text, audio_mfcc, tokenizer):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

    audio_tensor = torch.tensor(audio_mfcc, dtype=torch.float32).unsqueeze(0)

    return input_ids, attention_mask, token_type_ids, audio_tensor


# Electra + BiLSTM 멀티모달 모델 정의
class MultiModalBiLSTMClassifier(nn.Module):
    def __init__(self, num_major, num_urg, freeze_electra=True):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained("beomi/KcELECTRA-base")
        hidden_size = self.text_encoder.config.hidden_size  # 768

        if freeze_electra:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.audio_lstm = nn.LSTM(
            input_size=40,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.audio_bn = nn.BatchNorm1d(256)
        self.audio_dropout = nn.Dropout(0.3)

        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_size + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.major_head = nn.Linear(512, num_major)
        self.urg_head   = nn.Linear(512, num_urg)

    def forward(self, input_ids, attention_mask, token_type_ids, audio):
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_vec = text_out.last_hidden_state[:, 0, :]

        audio_out, _ = self.audio_lstm(audio)
        audio_vec = audio_out[:, -1, :]
        audio_vec = self.audio_bn(audio_vec)
        audio_vec = self.audio_dropout(audio_vec)

        fused = torch.cat([cls_vec, audio_vec], dim=1)
        fused = self.fusion_fc(fused)

        major_logits = self.major_head(fused)
        urg_logits   = self.urg_head(fused)

        return major_logits, urg_logits


# 멀티모달 모델 로드
num_major_classes = 4
num_urg_classes = 3

model = MultiModalBiLSTMClassifier(
    num_major=num_major_classes,
    num_urg=num_urg_classes,
    freeze_electra=False
)
model.load_state_dict(torch.load(
    "./model/finetuned_electra_bilstm.pt",
    map_location=device
))
model.to(device)
model.eval()


# 위치 추출
def extract_location_with_llm(stt_text):
    prompt = f"""
다음 STT에서 실제 '위치 정보'만 모두 추출하세요.
출력 형식은 반드시 JSON 배열만 출력하세요.

STT:
{stt_text}
"""
    response = location_model.generate_content(prompt)
    text = response.text.strip()

    try:
        return json.loads(text)
    except:
        return []


# 요약 생성
def summarize_with_llm(stt_text, major, urgency, locations):
    if isinstance(locations, list):
        location_text = ", ".join(locations) if locations else "없음"

    prompt = f"""
[STT]
{stt_text}

[모델 분석 결과]
- 상황: {major}
- 긴급도: {urgency}
- 위치: {location_text}

아래 형식으로 119 상황 요약을 생성하세요:

=== 상황 요약 ===
(핵심 사건 2~4줄)

=== 대응 필요성 판단 ===
(왜 출동해야 하는지, 위험성 중심)

=== 출동 요약 메시지 ===
(상황실 전달용 한 문장)
"""
    response = summary_model.generate_content(prompt)
    return response.text.strip()


# 전체 파이프라인 실행
def predict_pipeline(wav_path):
    print("🎤 STT 변환 중...")
    stt_text = run_stt(wav_path)

    print("📍 위치 정보 추출 중...")
    locations = extract_location_with_llm(stt_text)

    print("🎧 MFCC 추출 중...")
    audio_mfcc = extract_mfcc_sequence(wav_path)

    print("⚙ 입력 구성 중...")
    input_ids, attention_mask, token_type_ids, audio_tensor = make_inputs(
        stt_text, audio_mfcc, tokenizer
    )

    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    audio_tensor   = audio_tensor.to(device)

    print("🤖 상황 분류 모델 추론 중...")
    with torch.no_grad():
        major_logit, urg_logit = model(
            input_ids, attention_mask, token_type_ids, audio_tensor
        )

    major_idx = torch.argmax(major_logit, 1).item()
    urg_idx   = torch.argmax(urg_logit, 1).item()

    major_map = {0: "구급", 1: "구조", 2: "화재", 3: "기타"}
    urg_map   = {0: "하", 1: "중", 2: "상"}

    major = major_map.get(major_idx, "N/A")
    urgency = urg_map.get(urg_idx, "N/A")

    print("🧠 종합 요약 생성 중...")
    summary = summarize_with_llm(stt_text, major, urgency, locations)

    return {
        "text": stt_text,
        "major": major,
        "urgency": urgency,
        "locations": locations,
        "llm_summary": summary
    }


# 출력 포맷
def print_result(result):
    print("\n=================== 🆘 119 신고 분석 결과 ===================")

    print("\n📄 STT 추출 내용")
    print("-" * 60)
    print(result["text"])

    print("\n🧭 모델 분류 결과")
    print("-" * 60)
    print(f"• 상황 분류: {result['major']}")
    print(f"• 긴급도:   {result['urgency']}")

    print("\n🧠 LLM 종합 요약")
    print("-" * 60)
    print(result["llm_summary"])

    print("\n============================================================\n")


if __name__ == "__main__":
    # result = predict_pipeline("./demo/2/64dd752b1ef84058319a7fd1_20230212123359.wav")
    # print_result(result)
    
    result = predict_pipeline("./demo/2/64d9fdff3e12da15ae3a5940_20230211201601.wav")
    print_result(result)
    
    # result = predict_pipeline("./demo/2/6551fb0dd9c67ad7fa18a6fc_20220228.wav")
    # print_result(result)
