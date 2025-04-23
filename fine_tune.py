# 1. IMPORTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderCache, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import json, re, os, warnings, logging
from datetime import datetime

# 2. LOGGING CONFIGURATION : ghi lại thông tin trong quá trình chạy chương trình  theo dõi training
log_dir = "training/logs"  #tạo thư mục
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler() #In thông tin log ra console ngay khi chạy
    ]
)

# 3. PREPROCESSING FUNCTIONS  làm sạch văn bản đầu vào trước khi đưa vào model
def preprocess_input(text):
    """Xử lý text đầu vào: lowercase và loại bỏ ký tự đặc biệt"""
    text = text.strip().lower()
    #strip(): Loại bỏ khoảng trắng đầu/cuối chuỗi.
    #lower(): Chuyển tất cả chữ cái thành chữ thường
    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', '', text)
    return text
#Giúp văn bản đồng nhất và sạch hơn, dễ học hơn cho mô hình.
# 4. DATASET CLASS
class ChatbotDataset(Dataset):
    """Custom Dataset cho chatbot"""
    def __init__(self, data, tokenizer, max_length=128): #Hàm khởi tạo
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data) #Trả về tổng số lượng mẫu trong dataset.

    def __getitem__(self, idx): # 
        item = self.data[idx]
        input_text = f"chat: {item['input']}"
        target_text = item['output']
 # Lấy câu hỏi và câu trả lời từ dữ liệu, định dạng câu hỏi với prefix "chat: " để mô hình hiểu rõ mục đích đầu vào.


        # Tokenize input và target
        #Token hóa đầu vào và đầu ra
        #Token hóa là quá trình chuyển đổi văn bản thành các token (đơn vị ngữ nghĩa nhỏ hơn) mà mô hình có thể hiểu được.
        input_encodings = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        target_encodings = self.tokenizer(
            target_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        # Xử lý encodings
        input_ids = input_encodings['input_ids'].squeeze() #ID của token đầu vào.
        attention_mask = input_encodings['attention_mask'].squeeze() #mask giúp model phân biệt token thật và padding.
        labels = target_encodings['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100 # Đặt giá trị padding thành -100 để không tính vào loss.

        return { #Trả về một từ điển chứa các thông tin cần thiết cho mô hình.
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

 def evaluate_model(model, dataloader, device):
    """Đánh giá model trên validation set"""
    #Tóm tắt nhanh:
# Hàm này chạy model trên tập validation, tính average loss để đánh giá xem mô hình học tốt đến đâu sau mỗi epoch.
# Dùng torch.no_grad() để tắt việc tính gradient — rất cần thiết khi không training.
# Dùng .eval() để đảm bảo model hoạt động đúng chế độ inference.
    
    # Đặt model về chế độ evaluation (không dùng dropout, batchnorm,...)
    model.eval()

    total_val_loss = 0  # Biến dùng để cộng dồn loss trong quá trình đánh giá

    # Không tính toán gradient trong khi đánh giá → tiết kiệm bộ nhớ và tăng tốc
    with torch.no_grad():
        for batch in dataloader:
            # Đưa dữ liệu batch lên GPU (hoặc CPU tùy thiết bị)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass: tính toán output và loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # Cộng dồn loss vào tổng loss
            total_val_loss += outputs.loss.item()
    
    # Trả về loss trung bình trên toàn bộ tập validation
    return total_val_loss / len(dataloader)

def main():
    # 6.1 Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Kiểm tra xem GPU có sẵn không, nếu không dùng CPU
    print(f"\n🖥️ Đang sử dụng: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 2  # Số lượng mẫu trong mỗi batch
    MAX_LENGTH = 64  # Độ dài tối đa của chuỗi đầu vào và đầu ra
    NUM_EPOCHS = 5  # Số epoch huấn luyện
    
    # Directories
    cache_dir = "model_cache"  # Thư mục cache model đã tải
    results_dir = "training/results"  # Thư mục lưu kết quả
    os.makedirs(results_dir, exist_ok=True)  # Tạo thư mục kết quả nếu chưa có

    # 6.2 Model Loading
    print("\n1️⃣ Đang tải model từ cache...")
    try:
        # Tải tokenizer và model từ cache
        tokenizer = AutoTokenizer.from_pretrained(f"{cache_dir}/tokenizer", local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            f"{cache_dir}/model",  # Tải model đã huấn luyện từ cache
            local_files_only=True,
            use_cache=False  # Tắt sử dụng cache khi tải model
        ).to(device)  # Đưa model lên thiết bị (GPU/CPU)
        print("✅ Tải model thành công!")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return

    # 6.3 Data Loading
    print("\n2️⃣ Đang tải dữ liệu...")
    try:
        # Tải dữ liệu huấn luyện từ file JSON
        with open('training/data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Đã tải {len(data)} mẫu dữ liệu")
    except FileNotFoundError:
        print("❌ Không tìm thấy file data.json!")
        return

    # 6.4 Data Preparation
    print("\n3️⃣ Chuẩn bị dữ liệu...")
    # Tạo dataset từ dữ liệu đã tải và tokenizer
    dataset = ChatbotDataset(data, tokenizer, max_length=MAX_LENGTH)
    # Chia tập dữ liệu thành train và validation
    train_size = int(0.9 * len(dataset))  # 90% cho training
    val_size = len(dataset) - train_size  # 10% cho validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Dataloader cho train và validation
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,  # Số lượng mẫu trong mỗi batch
        shuffle=True,  # Shuffle dữ liệu
        num_workers=2,  # Sử dụng 2 worker để load dữ liệu song song
        pin_memory=True  # Dữ liệu sẽ được lưu trong bộ nhớ pin để truy cập nhanh hơn
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        pin_memory=True
    )

    # 6.5 Optimizer & Scheduler Setup
    # Khởi tạo optimizer và learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # AdamW optimizer
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=50,  # Số bước warmup
        num_training_steps=len(train_dataloader) * NUM_EPOCHS  # Tổng số bước huấn luyện
    )

    # 6.6 Training Loop
    print(f"\n4️⃣ Bắt đầu training...")
    print(f"📊 Train: {len(train_dataset)} mẫu, Validation: {len(val_dataset)} mẫu")
    print(f"📈 Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

    best_val_loss = float('inf')  # Khởi tạo giá trị loss tối ưu ban đầu
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()  # Chuyển model sang chế độ training
        total_loss = 0  # Khởi tạo tổng loss
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for batch in progress_bar:
            # Forward pass (tính output từ batch)
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            
            # Tính loss và làm backpropagation
            loss = outputs.loss
            total_loss += loss.item()  # Cộng dồn loss
            loss.backward()  # Backpropagation

            # Optimization step (cập nhật tham số của model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()  # Cập nhật tham số
            scheduler.step()  # Cập nhật learning rate
            optimizer.zero_grad()  # Reset gradient

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})  # Hiển thị loss trong progress bar

        # Validation phase
        val_loss = evaluate_model(model, val_dataloader, device)  # Đánh giá loss trên tập validation
        print(f"\n📊 Epoch {epoch+1}: Train Loss = {total_loss/len(train_dataloader):.4f}, Val Loss = {val_loss:.4f}")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("💾 Lưu model tốt nhất...")
            model.save_pretrained(f"{results_dir}/best_model")
            tokenizer.save_pretrained(f"{results_dir}/best_model")

    # 6.7 Model Testing
    print("\n5️⃣ Test model...")
    test_inputs = [
        "Xin chào!",  # Test case 1
        "2 + 2 bằng mấy?",  # Test case 2
        "Hôm nay thời tiết thế nào?"  # Test case 3
    ]

    model.eval()  # Chuyển model sang chế độ inference
    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        processed_input = preprocess_input(test_input)  # Tiền xử lý đầu vào
        inputs = tokenizer(
            f"chat: {processed_input}", 
            return_tensors="pt",  # Chuyển đổi thành tensor PyTorch
            max_length=64,  # Đặt độ dài tối đa cho chuỗi
            padding=True,  # Padding nếu cần
            truncation=True  # Cắt bớt nếu quá độ dài
        ).to(device)

        # Sinh câu trả lời từ mô hình
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=64,  # Độ dài tối đa cho câu trả lời
                num_beams=3,  # Beam search với 3 beam
                early_stopping=True  # Dừng sớm khi đã tìm được câu trả lời tốt
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Giải mã câu trả lời
        print(f"Output: {response}")

    print("\n✨ Training hoàn tất!")

# 7. ENTRY POINT
if __name__ == "__main__":
    main()
