import pandas as pd

class FeatureEngineer:
    #Verilen bilgilerden yeni bilgiler(özellikler) türetilmiştir.
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:  #yeni bir dataframe oluşturuyoruz ve oluşturulan bilgiler burada saklanıyor.
        df = df.copy() #ana veride değişiklik olmamamsı için kopya oluşturmak lazım.

        # 1. Title (Unvan) Türetme Yolcu isminden (Mr., Mrs., Miss. vb.) unvanları ayıklıyoruz
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Nadir görülen unvanları 'Rare' kategorisinde topluyoruz çünkü az olduğunda makinenin kafasını karıştırabilir.
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df['Title'] = df['Title'].replace(rare_titles, 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        #başka dilllerdeki kelimleri ingilizceye çeviriyoruz.
        # 2. FamilySize (Aile Büyüklüğü) ve IsAlone (Yalnız mı?)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # 3. Has_Cabin (Kabin Bilgisi Var Mı?)
        # Cabin sütununun %77'si boş olduğu için 1 (var) / 0 (yok) değişkenine dönüştürüyoruz
        df['Has_Cabin'] = df['Cabin'].notnull().astype(int)

        print("[INFO] Feature Engineering işlemleri başarıyla uygulandı.")
        return df
if __name__ == "__main__":
    # Test amaçlı basit bir dataframe
    sample_df = pd.DataFrame({
        'Name': ['Braund, Mr. Owen Harris', 'Heikkinen, Llle. Laina'],
        'SibSp': [1, 0],
        'Parch': [0, 0],
        'Cabin': [None, 'C85']
    })
    fe = FeatureEngineer()
    res = fe.create_features(sample_df)
    print(res[['Title', 'FamilySize', 'IsAlone', 'Has_Cabin']])
#kodun çalşışabilirliği test edilşmiştir.