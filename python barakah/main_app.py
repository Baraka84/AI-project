import os
import librosa
import numpy as np
import joblib
import customtkinter as ctk
from tkinter import filedialog, messagebox
from sklearn.svm import SVC

# --- إعداد المسارات بشكل مرن جداً ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'voice_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, "data")
HUMAN_PATH = os.path.join(DATA_PATH, "human")
AI_PATH = os.path.join(DATA_PATH, "ai")

# إنشاء المجلدات تلقائياً إذا كانت مفقودة
for path in [HUMAN_PATH, AI_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

def extract_features(file_path):
    try:
        # قراءة الملف - يدعم صيغ الواتساب وأي صيغة أخرى
        audio, sample_rate = librosa.load(file_path, duration=3)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"خطأ في الملف {file_path}: {e}")
        return None

def train_now():
    features, labels = [], []
    
    # مابين الأقواس هي الصيغ التي سيبحث عنها البرنامج
    valid_formats = ('.wav', '.mp3', '.opus', '.m4a', '.aac')

    # قراءة ملفات البشر
    for file in os.listdir(HUMAN_PATH):
        if file.lower().endswith(valid_formats):
            f = extract_features(os.path.join(HUMAN_PATH, file))
            if f is not None:
                features.append(f)
                labels.append(0)

    # قراءة ملفات الذكاء الاصطناعي
    for file in os.listdir(AI_PATH):
        if file.lower().endswith(valid_formats):
            f = extract_features(os.path.join(AI_PATH, file))
            if f is not None:
                features.append(f)
                labels.append(1)

    if len(features) < 2: # يحتاج الموديل على الأقل ملفين للتعلم
        return False, "تحتاج لملفين على الأقل (واحد في كل مجلد) للتدريب."
    
    try:
        model = SVC(kernel='linear', probability=True)
        model.fit(features, labels)
        joblib.dump(model, MODEL_PATH)
        return True, f"تم التدريب بنجاح على {len(features)} ملف!"
    except Exception as e:
        return False, str(e)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Audio Analytics System")
        self.geometry("1000x400")
        
        ctk.CTkLabel(self, text="نظام كشف الأصوات المزيفة", font=("Arial", 50)).pack(pady=50)
        
        # زر التدريب
        self.btn_train = ctk.CTkButton(
            self, 
            text="1. تدريب الموديل (تحديث)", 
            command=self.do_train, 
            fg_color="green",
            width=300,   # العرض بالبكسل
            height=60,   # الطول بالبكسل
            font=("Arial", 18, "bold") # تكبير الخط أيضاً ليتناسب مع الزر
        )
        self.btn_train.pack(pady=20) # زيادة المسافة الرأسية
        
        # زر الفحص
        self.btn_test = ctk.CTkButton(
            self, 
            text="2. فحص ملف صوتي", 
            command=self.do_test,
            width=300,   # نفس العرض ليكون التنسيق متناسق
            height=60,   # نفس الطول
            font=("Arial", 18, "bold")
        )
        self.btn_test.pack(pady=20)

    def do_train(self):
        success, msg = train_now()
        if success:
            messagebox.showinfo("نجاح", msg)
        else:
            messagebox.showwarning("تنبيه", msg)

    def do_test(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("خطأ", "يجب الضغط على زر التدريب أولاً!")
            return
        
        path = filedialog.askopenfilename()
        if path:
            feat = extract_features(path)
            if feat is not None:
                model = joblib.load(MODEL_PATH)
                res = model.predict([feat])
                result_text = "صوت بشري حقيقي" if res[0] == 0 else "صوت ذكاء اصطناعي"
                messagebox.showinfo("النتيجة", f"هذا الصوت هو: {result_text}")

if __name__ == "__main__":
    App().mainloop()