from pydub import AudioSegment
import os

# حدد مكان المجلد اللي فيه أصوات الواتساب
input_folder = "data/human" 

for filename in os.listdir(input_folder):
    if filename.endswith(".opus") or filename.endswith(".aac") or filename.endswith(".m4a"):
        print(f"جاري تحويل: {filename}")
        
        # تحميل الملف
        audio = AudioSegment.from_file(os.path.join(input_folder, filename))
        
        # تغيير الاسم ليكون .wav
        new_filename = os.path.splitext(filename)[0] + ".wav"
        
        # حفظ الملف بالصيغة الجديدة
        audio.export(os.path.join(input_folder, new_filename), format="wav")
        print(f"تم بنجاح حفظ: {new_filename}")