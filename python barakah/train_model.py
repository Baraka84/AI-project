import os
import librosa
import numpy as np
from sklearn.svm import SVC
import joblib

# 1. دالة استخراج البصمة الصوتية
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"خطأ في قراءة الملف {file_path}: {e}")
        return None

# 2. تحديد مسارات المجلدات (تأكد أنها بنفس اسم المجلدات عندك)
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "data")
human_folder = os.path.join(data_path, "human")
ai_folder = os.path.join(data_path, "ai")

features = []
labels = []

print("--- جاري قراءة الملفات وتدريب الموديل... انتظر قليلاً ---")

# معالجة ملفات البشر (Label = 0)
if os.path.exists(human_folder):
    for file in os.listdir(human_folder):
        if file.endswith(('.wav', '.mp3', '.opus', '.m4a')):
            feat = extract_features(os.path.join(human_folder, file))
            if feat is not None:
                features.append(feat)
                labels.append(0)

# معالجة ملفات الذكاء الاصطناعي (Label = 1)
if os.path.exists(ai_folder):
    for file in os.listdir(ai_folder):
        if file.endswith(('.wav', '.mp3', '.opus', '.m4a')):
            feat = extract_features(os.path.join(ai_folder, file))
            if feat is not None:
                features.append(feat)
                labels.append(1)

# 3. حفظ الموديل في ملف pkl
if len(features) > 0:
    model = SVC(kernel='linear', probability=True)
    model.fit(features, labels)
    # هذه الخطوة هي التي ستنشئ الملف الذي تحتاجه الواجهة
    joblib.dump(model, 'voice_model.pkl')
    print(f"✅ تم بنجاح! تم إنشاء ملف voice_model.pkl بناءً على {len(features)} ملف صوتي.")
else:
    print("❌ خطأ: لم يتم العثور على أي ملفات صوتية في مجلدات data.")