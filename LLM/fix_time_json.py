import json

def format_time(seconds_str):
    """تبدیل ثانیه به فرمت دقیقه:ثانیه"""
    total_seconds = int(seconds_str)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"

def process_and_save_chunk_times(input_path, output_path):
    # خواندن داده‌ها از فایل اصلی
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # مدیریت داده (پشتیبانی از لیست یا یک آبجکت تکی)
    items = data if isinstance(data, list) else [data]
    
    # اعمال تغییرات زمانی روی هر آیتم
    for item in items:
        if "start_time" in item:
            item["start_time"] = format_time(item["start_time"])
        if "end_time" in item:
            item["end_time"] = format_time(item["end_time"])
            
    # ذخیره داده‌های به‌روزرسانی‌شده در فایل جدید
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# process_and_save_chunk_times("input_data.json", "updated_data.json")
# برای اجرای کد، مسیر فایل خود را به تابع بدهید:
process_and_save_chunk_times("all_processed_chunks_with_reasoning(2).json", "updated_chunks.json")