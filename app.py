import streamlit as st
import pickle
import os
import re
from functools import lru_cache
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Định nghĩa các biến đường dẫn
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(MODEL_DIR, "kneserney_trigram_model.pkl")
SYLLABLES_PATH = os.path.join(DATA_DIR, "vn_syllables.txt")

# Hàm loại bỏ dấu tiếng Việt
def remove_vn_accent(word: str) -> str:
    word = word.lower()
    word = re.sub(r'[áàảãạăắằẳẵặâấầẩẫậ]', 'a', word)
    word = re.sub(r'[éèẻẽẹêếềểễệ]', 'e', word)
    word = re.sub(r'[óòỏõọôốồổỗộơớờởỡợ]', 'o', word)
    word = re.sub(r'[íìỉĩị]', 'i', word)
    word = re.sub(r'[úùủũụưứừửữự]', 'u', word)
    word = re.sub(r'[ýỳỷỹỵ]', 'y', word)
    word = re.sub(r'đ', 'd', word)
    return word

# Hàm sinh các biến thể có dấu của một từ không dấu
def gen_accents_word(word: str, syllables_path: str = SYLLABLES_PATH) -> set[str]:
    normalized_input_word = word.lower()
    word_no_accent = remove_vn_accent(normalized_input_word)
    all_accent_word = {normalized_input_word}
    if not os.path.exists(syllables_path):
        return all_accent_word
    try:
        with open(syllables_path, 'r', encoding='utf-8') as f:
            for w_line in f.read().splitlines():
                w_line_lower = w_line.lower()
                if remove_vn_accent(w_line_lower) == word_no_accent:
                    all_accent_word.add(w_line_lower)
    except Exception:
        pass
    return all_accent_word

# Mô hình n-gram tuỳ chỉnh
class CustomNGramModel:
    def __init__(self, model_dict):
        self.order = model_dict["n"]
        self.vocab = set(model_dict["vocab"])
        self.ngram_counts = model_dict["ngram_counts"]
        self.context_counts = model_dict["context_counts"]
        self.alpha = 1e-10
    @lru_cache(maxsize=10000)
    def logscore(self, word, context=None):
        import math
        if context is None:
            context = tuple()
        if len(context) > self.order - 1:
            context = context[-(self.order - 1):]
        if len(context) < self.order - 1:
            context = tuple(["<s>"] * ((self.order - 1) - len(context))) + tuple(context)
        ngram = context + (word,)
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)
        if context_count == 0:
            return math.log(self.alpha)
        prob = (ngram_count + self.alpha) / (context_count + self.alpha * len(self.vocab))
        return math.log(prob)

# Cache kết quả cho gen_accents_word
def cached_gen_accents_word(word, syllables_path):
    return gen_accents_word(word, syllables_path)

# Hàm beam search dự đoán câu có dấu
def beam_search_predict_accents(text_no_accents: str, model, k: int = 3,
                               syllables_path: str = SYLLABLES_PATH,
                               detokenizer=TreebankWordDetokenizer()) -> list[tuple[str, float]]:
    if isinstance(model, dict) and "n" in model and "vocab" in model:
        model = CustomNGramModel(model)
    words = text_no_accents.lower().split()
    if not words:
        return []
    sequences = []
    for idx, word_no_accent in enumerate(words):
        possible_accented_words = cached_gen_accents_word(word_no_accent, syllables_path)
        if not possible_accented_words:
            possible_accented_words = {word_no_accent}
        if idx == 0:
            sequences = [([word], 0.0) for word in possible_accented_words]
            continue
        all_new_sequences = []
        for seq_words, seq_score in sequences:
            context = seq_words[-(model.order - 1):] if model.order > 1 else []
            for next_word in possible_accented_words:
                try:
                    score_addition = model.logscore(next_word, tuple(context))
                    new_seq = (seq_words + [next_word], seq_score + score_addition)
                    all_new_sequences.append(new_seq)
                except Exception:
                    continue
        if not all_new_sequences:
            sequences = [(seq[0] + [word_no_accent], seq[1] - 1000) for seq in sequences[:1] or [([],0)]]
            continue
        all_new_sequences.sort(key=lambda x: x[1], reverse=True)
        sequences = all_new_sequences[:k]
    results = [(detokenizer.detokenize(seq_words), score) for seq_words, score in sequences]
    return results

# Giao diện Streamlit
st.set_page_config(layout="wide", page_title="Thêm Dấu Tiếng Việt")

st.title("Ứng dụng Thêm Dấu Tiếng Việt Tự Động")
st.markdown("""
    Nhập một câu tiếng Việt không dấu vào ô bên dưới và nhấn Enter.
    Ứng dụng sẽ dự đoán và hiển thị các câu có dấu khả năng cao nhất.
""")

@st.cache_resource
def load_model_cached(): # Đổi tên hàm để tránh xung đột với biến model
    with st.spinner("Đang tải model, vui lòng chờ..."):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)

try:
    model_data = load_model_cached() # Gán vào biến mới

    st.sidebar.header("Nhập liệu")
    user_input = st.sidebar.text_input("Nhập câu tiếng Việt không dấu:", "", key="user_input")

    if user_input:
        with st.spinner("Đang xử lý dự đoán..."):
            predictions = beam_search_predict_accents(user_input, model_data, k=5, syllables_path=SYLLABLES_PATH)

        if predictions:
            st.subheader("Kết quả dự đoán:")
            
            best_prediction_col, other_predictions_col = st.columns([2,3])

            with best_prediction_col:
                st.success(f"**Câu dự đoán tốt nhất:**\n ## {predictions[0][0]}")
                st.caption(f"(Điểm: {predictions[0][1]:.4f})")

            with other_predictions_col:
                if len(predictions) > 1:
                    st.write("#### Các dự đoán khác:")
                    for i, (sent, score) in enumerate(predictions[1:], start=1):
                        st.info(f"{i+1}. **'{sent}'** (Điểm: {score:.4f})")
        else:
            st.warning("Không thể dự đoán được câu có dấu cho đầu vào này.")
    else:
        st.info("Vui lòng nhập câu vào thanh bên để bắt đầu.")

except FileNotFoundError as fnf_error:
    st.error(f"Lỗi không tìm thấy file: {str(fnf_error)}")
    st.error(f"Vui lòng kiểm tra lại đường dẫn đến file model ({MODEL_PATH}) và file âm tiết ({SYLLABLES_PATH}).")
    st.warning("Đảm bảo rằng các thư mục 'models' và 'data' tồn tại trong cùng thư mục với file app.py và chứa các file cần thiết.")
except Exception as e:
    st.error(f"Có lỗi không mong muốn xảy ra: {str(e)}")
    st.error("Vui lòng kiểm tra lại đường dẫn đến file model và file âm tiết, hoặc thử khởi động lại ứng dụng.")

# Thêm mã để kiểm tra đường dẫn
# st.sidebar.write(f"Model path: {MODEL_PATH}")
# st.sidebar.write(f"Model exists: {os.path.exists(MODEL_PATH)}")
# st.sidebar.write(f"Syllables path: {SYLLABLES_PATH}")
# st.sidebar.write(f"Syllables exists: {os.path.exists(SYLLABLES_PATH)}")

# # Thêm nút để xóa cache
# if st.sidebar.button("Xóa cache và tải lại model"):
#     st.cache_resource.clear()
#     st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.info("© Phan Tấn Tài & Phan Nhật Trường")
