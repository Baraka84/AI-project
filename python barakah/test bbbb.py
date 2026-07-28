import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 1. دالة استخراج البصمة الصوتية
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"خطأ في الملف {file_path}: {e}")
        return None

# 2. تحديد المسارات
base_path = os.path.dirname(os.path.abspath(__file__))
human_folder = os.path.join(base_path, "data", "human")
ai_folder = os.path.join(base_path, "data", "ai")

features = []
labels = []

print("--- جاري تحليل الأصوات، انتظر قليلاً ---")

# 3. معالجة أصوات البشر (Label = 0)
if os.path.exists(human_folder):
    for file in os.listdir(human_folder):
        if file.endswith(('.wav', '.mp3', '.opus', '.m4a')):
            path = os.path.join(human_folder, file)
            feat = extract_features(path)
            if feat is not None:
                features.append(feat)
                labels.append(0)

# 4. معالجة أصوات الذكاء الاصطناعي (Label = 1)
if os.path.exists(ai_folder):
    for file in os.listdir(ai_folder):
        if file.endswith(('.wav', '.mp3', '.opus', '.m4a')):
            path = os.path.join(ai_folder, file)
            feat = extract_features(path)
            if feat is not None:
                features.append(feat)
                labels.append(1)

# 5. مرحلة التدريب والنتيجة
if len(features) > 0:
    print(f"تم العثور على {len(features)} ملفات. جاري تدريب الذكاء الاصطناعي...")
    model = SVC(kernel='linear')
    model.fit(features, labels)
    print("--- تم التدريب بنجاح! برنامجك الآن جاهز للتمييز ---")
else:
    print("تنبيه: لم يتم العثور على أي ملفات صوتية في المجلدات!")