import os
import pandas as pd
from typing import Tuple, Optional

class DataLoader:
    def __init__(self, raw_data_dir: str ="data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        data/raw altındaki train.csv ve test.csv dosyalarını okur.Klasör içindeki verileri okumak için yazdım bu fonskiyonu.
        geriye döndüreceği veriler iki adet pandas data verisidir.
        -----Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) vb.
        """
        train_path = os.path.join(self.raw_data_dir, "train.csv") #train dosyasının veri yolu bu değişkene atılıyor.
        test_path = os.path.join(self.raw_data_dir, "test.csv") #test dosyasını veri yolu bu değişkene atılıyor.

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Hata: train.csv veya test.csv '{self.raw_data_dir}' dizininde bulunamamıştırm! "
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"--BİLGİ-- Ham veriler başarıyla yüklenmiştir:")
        print(f"        - Train Seti: {train_df.shape}") #train setinin büyüklüğü
        print(f"        - Test Seti : {test_df.shape}")  #test setinin büyüklüğü

        return train_df, test_df

    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        #işlemden geçen verileri kaydetmek için yazılmış method
        os.makedirs(self.processed_data_dir, exist_ok=True)
        train_path = os.path.join(self.processed_data_dir, "train_processed.csv")
        test_path = os.path.join(self.processed_data_dir, "test_processed.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Bilgi(İşlenmiş veriler) '{self.processed_data_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    # Modülün bağımsız test edilebilirliği için:Kodun ayrı olarak çalıştığı burda test edilmektedir.
    loader = DataLoader() #nesne oluşturma
    try:
        train, test = loader.load_raw_data()
        print("\nTrain İlk 3 Satır:")
        print(train.head(3))
        print(test.head(3))
    except Exception as e:
        print(e)