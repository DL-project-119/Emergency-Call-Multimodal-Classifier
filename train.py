import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

BASE = "./data"
MODEL_NAME = "beomi/KcELECTRA-base"

# 데이터 로드
train_df = pd.read_csv(f"{BASE}/text/train_final.csv")
valid_df = pd.read_csv(f"{BASE}/text/valid_final.csv")
test_df  = pd.read_csv(f"{BASE}/text/test_final.csv")

X_train_audio = np.load(f"{BASE}/audio/X_audio_train_seq.npy")
X_valid_audio = np.load(f"{BASE}/audio/X_audio_valid_seq.npy")
X_test_audio  = np.load(f"{BASE}/audio/X_audio_test_seq.npy")

print("train_df:", train_df.shape, "X_train_audio:", X_train_audio.shape)

num_major_classes = train_df["major_label"].nunique()
num_urg_classes   = train_df["urgency_label"].nunique()


# 오디오 피처 스케일링
do_scaling = True

if do_scaling:
    N_train, T, F = X_train_audio.shape
    scaler = StandardScaler()

    scaler.fit(X_train_audio.reshape(-1, F))

    X_train_audio = scaler.transform(X_train_audio.reshape(-1, F)).reshape(N_train, T, F)
    X_valid_audio = scaler.transform(X_valid_audio.reshape(-1, F)).reshape(X_valid_audio.shape)
    X_test_audio  = scaler.transform(X_test_audio.reshape(-1, F)).reshape(X_test_audio.shape)


# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset 정의
class MultiModalSeqDataset(Dataset):
    def __init__(self, df, audio_features, tokenizer, max_len=512):
        self.df = df.reset_index(drop=True)
        self.audio = audio_features
        self.tok = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        major = int(row["major_label"])
        urg   = int(row["urgency_label"])
        audio = torch.tensor(self.audio[idx], dtype=torch.float32)
        
        enc = self.tok(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze()
        attention = enc["attention_mask"].squeeze()
        token_type = enc.get("token_type_ids", torch.zeros_like(input_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "token_type_ids": token_type,
            "audio": audio,
            "major_label": torch.tensor(major),
            "urgency_label": torch.tensor(urg),
        }

    def __len__(self):
        return len(self.df)


train_loader = DataLoader(MultiModalSeqDataset(train_df, X_train_audio, tokenizer), batch_size=16, shuffle=True)
valid_loader = DataLoader(MultiModalSeqDataset(valid_df, X_valid_audio, tokenizer), batch_size=32)
test_loader  = DataLoader(MultiModalSeqDataset(test_df,  X_test_audio,  tokenizer), batch_size=32)


# 모델 정의
class MultiModalBiLSTMClassifier(nn.Module):
    def __init__(self, num_major, num_urg, freeze_electra=True):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.text_encoder.config.hidden_size  # 768

        if freeze_electra:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.audio_lstm = nn.LSTM(
            input_size=40, hidden_size=128,
            batch_first=True, bidirectional=True
        )
        self.audio_bn = nn.BatchNorm1d(256)
        self.audio_drop = nn.Dropout(0.3)

        self.fusion = nn.Sequential(
            nn.Linear(hidden + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.major_head = nn.Linear(512, num_major)
        self.urg_head   = nn.Linear(512, num_urg)

    def forward(self, input_ids, attention, token_type, audio):
        txt_out = self.text_encoder(input_ids, attention_mask=attention, token_type_ids=token_type)
        cls_vec = txt_out.last_hidden_state[:, 0, :]

        audio_out, _ = self.audio_lstm(audio)
        audio_vec = audio_out[:, -1, :]
        audio_vec = self.audio_bn(audio_vec)
        audio_vec = self.audio_drop(audio_vec)

        fused = self.fusion(torch.cat([cls_vec, audio_vec], dim=1))

        return self.major_head(fused), self.urg_head(fused)


# 평가 함수
def evaluate(model, loader, show_matrix=False):
    model.eval()
    preds_m, preds_u = [], []
    trues_m, trues_u = [], []

    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            tok = batch["token_type_ids"].to(device)
            audio = batch["audio"].to(device)

            m_logit, u_logit = model(inp, att, tok, audio)

            preds_m.extend(m_logit.argmax(1).cpu().numpy())
            preds_u.extend(u_logit.argmax(1).cpu().numpy())
            trues_m.extend(batch["major_label"].numpy())
            trues_u.extend(batch["urgency_label"].numpy())

    print("=== Major ===")
    print(classification_report(trues_m, preds_m, digits=4))
    print("=== Urgency ===")
    print(classification_report(trues_u, preds_u, digits=4))

    if show_matrix:
        cm_m = confusion_matrix(trues_m, preds_m)
        cm_u = confusion_matrix(trues_u, preds_u)
        print(cm_m)
        print(cm_u)

    return accuracy_score(trues_u, preds_u)


# 학습 함수
def train_model(model, train_loader, valid_loader, epochs=8, lr=3e-5, patience=2):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit_m = nn.CrossEntropyLoss()
    crit_u = nn.CrossEntropyLoss()

    best_acc, wait = 0, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        print(f"\n===== Epoch {epoch+1} / {epochs} =====")
        for batch in tqdm(train_loader):
            inp = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            tok = batch["token_type_ids"].to(device)
            audio = batch["audio"].to(device)

            major_y = batch["major_label"].to(device)
            urg_y   = batch["urgency_label"].to(device)

            m_logit, u_logit = model(inp, att, tok, audio)

            loss = crit_m(m_logit, major_y) + 2.5 * crit_u(u_logit, urg_y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()

        print(f"Train Loss: {total_loss/len(train_loader):.4f}")

        val_acc = evaluate(model, valid_loader)
        print(f"Valid Urgency Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            wait = 0
            print("- Best checkpoint updated")
        else:
            wait += 1
            if wait >= patience:
                print("- Early stopping")
                break

    model.load_state_dict(best_state)
    return model


# 학습 실행
if __name__ == "__main__":
    print("\n=== Baseline (freeze) Training ===")
    baseline = MultiModalBiLSTMClassifier(num_major_classes, num_urg_classes, freeze_electra=True)
    baseline = train_model(baseline, train_loader, valid_loader)

    print("\n=== Fine-tune Training ===")
    finetune = MultiModalBiLSTMClassifier(num_major_classes, num_urg_classes, freeze_electra=False)
    finetune = train_model(finetune, train_loader, valid_loader)

    print("\n=== Test Result: Baseline ===")
    evaluate(baseline, test_loader, show_matrix=True)

    print("\n=== Test Result: Fine-tuned ===")
    evaluate(finetune, test_loader, show_matrix=True)
