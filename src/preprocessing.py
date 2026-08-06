import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    
    def __init__(self):
        self.age_medians_by_title = {}
        self.fare_medians_by_pclass = {}
        self.embarked_mode = None
        self.age_scaler = StandardScaler()  # Age ölçekleyici eklendi

    def fit(self, train_df: pd.DataFrame):
        """
        Doldurma istatistiklerini (medyan, mod) ve Scaler ortalamasını SADECE Train setinden öğrenir.
        """
        # 1. Unvanlara göre yaş medyanlarını hesapla
        self.age_medians_by_title = train_df.groupby('Title')['Age'].median().to_dict()
        
        # 2. Pclass'a göre bilet ücreti medyanlarını hesapla
        self.fare_medians_by_pclass = train_df.groupby('Pclass')['Fare'].median().to_dict()
        
        # 3. Biniş limanı en çok tekrar eden değeri (Mod) bul
        self.embarked_mode = train_df['Embarked'].mode()[0]

        # 4. Age sütununun eksiklerini geçici doldurup Scaler'ı fit et (Sadece Train verisinden öğren)
        temp_age = train_df['Age'].copy()
        for title, median_age in self.age_medians_by_title.items():
            temp_age.loc[(temp_age.isnull()) & (train_df['Title'] == title)] = median_age
        temp_age = temp_age.fillna(temp_age.median())
        
        # StandardScaler'a (N, 1) boyutunda dizi vermek gerekir
        self.age_scaler.fit(temp_age.values.reshape(-1, 1))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Öğrenilen istatistikler ve eğitilmiş Scaler ile veriyi dönüştürür.
        """
        df = df.copy()

        # 1. Eksik Veri Doldurma (Imputation)
        # Embarked
        df['Embarked'] = df['Embarked'].fillna(self.embarked_mode)

        # Fare
        for pclass, median_fare in self.fare_medians_by_pclass.items():
            df.loc[(df['Fare'].isnull()) & (df['Pclass'] == pclass), 'Fare'] = median_fare

        # Age (Unvan bazlı medyan doldurma)
        for title, median_age in self.age_medians_by_title.items():
            df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = median_age
        df['Age'] = df['Age'].fillna(df['Age'].median())

        # 2. Age Sütununu Ölçekleme (StandardScaler)
        # fit() aşamasında öğrenilen mean ve std kullanılarak transform edilir
        df['Age'] = self.age_scaler.transform(df[['Age']])

        # 3. Fare için Log Dönüşümü (Çarpıklığı gidermek için)
        df['Fare_Log'] = np.log1p(df['Fare'])

        # 4. Categorical Encoding
        # Sex -> Binary (0 / 1)
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

        # Title & Embarked -> One-Hot Encoding
        df = pd.get_dummies(df, columns=['Title', 'Embarked'], drop_first=True, dtype=int)

        # 5. Gereksiz Sütunları Çıkarma
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        print("[INFO] Preprocessing ve Age scaling işlemleri başarıyla uygulandı.")
        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Train seti için fit ve transform adımlarını ardışık çalıştırır.
        """
        self.fit(train_df)
        return self.transform(train_df)