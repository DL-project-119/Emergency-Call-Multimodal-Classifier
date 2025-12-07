import os
import glob
import json
import pandas as pd
from tqdm import tqdm


STOPWORDS = [
    "[개인정보]", "119입니다.", "여보세요", "잠시만요", "지금", "그냥", "뭐야",
    "그러면", "그래요", "그러니까", "이제", "네", "예", "여기", "저기", "거기",
    "거기요", "저기요", "여기요", "네네", "네네네", "입니다.",
    "네.", "예 예 예.", "예예예.", "예예예, 예", "예, 알겠어요.",
    "차량 종이 뭐예요?", "차종 차종.", "차당 뭣이 뭐냐고."
]


# 텍스트 정제 함수
def clean_text(text: str) -> str:
    cleaned = text

    # 불용어 제거
    for sw in STOPWORDS:
        cleaned = cleaned.replace(sw, " ")

    # 특수문자 제거
    for ch in [",", ".", "?", "!", "\n"]:
        cleaned = cleaned.replace(ch, " ")

    # 공백 정리
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def process_json_file(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_id = os.path.basename(file_path).replace(".json", "")

        # 텍스트 합치기
        utterances = [u["text"] for u in data.get("utterances", [])]
        full_text = " ".join(utterances)
        cleaned = clean_text(full_text)

        # 레이블 가져오기
        return {
            "X": {
                "ID": file_id,
                "text": cleaned
            },
            "Y": {
                "ID": file_id,
                "disasterMedium": data.get("disasterMedium", "N/A"),
                "urgencyLevel": data.get("urgencyLevel", "N/A"),
                "sentiment": data.get("sentiment", "N/A")
            }
        }

    except Exception as e:
        tqdm.write(f"JSON 처리 오류: {file_path} — {e}")
        return None



# 전체 JSON -> CSV 변환 실행
def run_text_preprocessing(json_root, save_dir, folder_name):
    all_files = glob.glob(os.path.join(json_root, "**/*.json"), recursive=True)

    if not all_files:
        raise FileNotFoundError(f"{json_root} 에 JSON 없음.")

    X_list, Y_list = [], []

    for fp in tqdm(all_files):
        result = process_json_file(fp)
        if result:
            X_list.append(result["X"])
            Y_list.append(result["Y"])

    df_X = pd.DataFrame(X_list)
    df_Y = pd.DataFrame(Y_list)

    os.makedirs(save_dir, exist_ok=True)

    x_path = os.path.join(save_dir, f"X_features_{folder_name}_text.csv")
    y_path = os.path.join(save_dir, f"Y_labels_{folder_name}.csv")

    df_X.to_csv(x_path, index=False, encoding="utf-8-sig")
    df_Y.to_csv(y_path, index=False, encoding="utf-8-sig")

    print(f"\nX 저장: {x_path}")
    print(f"Y 저장: {y_path}")

    return df_X, df_Y


if __name__ == "__main__":
    run_text_preprocessing(
        json_root="./data/Training/Text/TL_구급",
        save_dir="./data/Training/Text/구급",
        folder_name="구급"
    )
