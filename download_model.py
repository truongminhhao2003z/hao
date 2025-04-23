from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login, HfFolder
import os

def download_model():
    # Login to Hugging Face
    try:
        token = "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"
        login(token=token)
        HfFolder.save_token(token)  # Save token for future use
        print("Đăng nhập thành công!")
    except Exception as e:
        print(f"Lỗi đăng nhập: {str(e)}")
        return

    cache_dir = "model_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Sửa tên model thành đúng
    model_name = "VietAI/vit5-base"  # Thay vì "VietAI/vimt5-base"
    print(f"Đang tải model {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        
        print("Đang lưu model vào cache...")
        tokenizer.save_pretrained(f"{cache_dir}/tokenizer")
        model.save_pretrained(f"{cache_dir}/model")
        print("Tải model thành công!")
        
    except Exception as e:
        print(f"Lỗi khi tải model: {str(e)}")

if __name__ == "__main__":
    download_model()