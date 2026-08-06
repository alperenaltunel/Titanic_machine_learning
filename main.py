import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Proje içindeki src modüllerini import ediyoruz
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.preprocessing import Preprocessor
from src.models import ModelFactory
from src.evaluate import ModelEvaluator

def run_pipeline():
    print("=" * 60)
    print("🚀 TITANIC ML PIPELINE BAŞLATILIYOR...")
    print("=" * 60)

    # 1. Veri Yükleme
    print("\n[1/5] Ham veriler yükleniyor...")
    loader = DataLoader(raw_data_dir="data/raw", processed_data_dir="data/processed")
    train_df, test_df = loader.load_raw_data()

    # 2. Öznitelik Mühendisliği (Feature Engineering)
    print("\n[2/5] Öznitelikler (Features) türetiliyor...")
    fe = FeatureEngineer()
    train_fe = fe.create_features(train_df)
    test_fe = fe.create_features(test_df)

    # 3. Ön İşleme (Preprocessing)
    print("\n[3/5] Ön işleme ve temizlik uygulanıyor...")
    prep = Preprocessor()
    train_proc = prep.fit_transform(train_fe)
    test_proc = prep.transform(test_fe)

    # İşlenmiş verileri diske kaydetme
    loader.save_processed_data(train_proc, test_proc)

    # 4. Veriyi Hazırlama ve Ölçekleme
    X = train_proc.drop(columns=['Survived'])
    y = train_proc['Survived']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 5. Modelleri Eğitme ve Değerlendirme
    print("\n[4/5] Modeller eğitiliyor ve değerlendiriliyor...")
    factory = ModelFactory()
    evaluator = ModelEvaluator()

    baseline_models = factory.get_models()
    param_grids = factory.get_param_grids()

    all_results = []

    # Baseline & Tuned Model Eğitimi
    for name, model in baseline_models.items():
        X_train = X if name in ["Decision Tree", "Random Forest", "Gradient Boosting"] else X_scaled
        
        # Baseline
        res_base = evaluator.evaluate_model(model, X_train, y, model_name=name)
        all_results.append(res_base)

        # Tuned
        grid = param_grids[name]
        _, res_tuned = evaluator.tune_and_evaluate(model, grid, X_train, y, model_name=name)
        all_results.append(res_tuned)

    # 6. Çıktıları Kaydetme
    print("\n[5/5] Sonuçlar ve raporlar kaydediliyor...")
    results_df = pd.DataFrame(all_results)
    
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/results_table.csv", index=False)

    print("\n" + "=" * 60)
    print("🎉 PIPELINE BAŞARIYLA TAMAMLANDI!")
    print("=" * 60)
    print("\nEn Başarılı 3 Model (F1-Score):")
    print(results_df.sort_values(by="F1-Score", ascending=False)[['Model', 'Val Accuracy', 'F1-Score', 'ROC-AUC']].head(3))

if __name__ == "__main__":
    run_pipeline()