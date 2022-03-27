""":arg

    input= yil, vites, motor gücü,km,motor hacmi
    output= fiyat

    denetimli makine öğrenmesi modeli uygulanabilir input ile output verileri
    karşılaştırıldığında input değişkenlerine bağlı bize fiyat bilgisini sunabilir.



"""


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import preprocessing

path='_sahibinden_.xlsx'

class adaboost:
    def __init__(self,index):
        self.index=index
        self.data1 = pd.read_excel(path, header=None, usecols=self.index)  # sütun isimleri hata veriyor column name ile devam
        array = self.data1.values
        array=np.delete(array,0) #okunan veriden başlığı çıkardım
        # scikit-learn'in Standart Ölçekleyicisi (birçok scikit-learn algoritması ve ML algoritması gibi) SADECE sayısal verileri kabul eder.
        # Bu nedenle, metin verilerinizden sayısal veriler yapmanız gerekir. Bunu, scikit-learn'in vektörleştiricilerinden birini - CountVectorizer, TfIdfVectorizer (önerilir) kullanarak yapabilirsiniz.

        vectorizer = CountVectorizer()
        XcountVectorizer = vectorizer.fit_transform(array.ravel())
        vectorizer.get_feature_names_out()
        self.array_XcountVectorizer = XcountVectorizer.toarray()

        X = self.array_XcountVectorizer[:, 0:1]
        Y = self.array_XcountVectorizer[:, 1]
        seed = 5
        kfold = KFold(n_splits=5, random_state=seed, shuffle=True)
        ADBclf = AdaBoostClassifier(n_estimators=100)
        self.results = cross_val_score(ADBclf, X, Y, cv=kfold)



    def show(self):
        return (print("column {} ".format(self.index) +": "+ str(self.results.mean())))

    def adaBoostRegresyon(self,n_features): #n_features değerleri otomatik alınabilir
        self.n_features=n_features
        X, Y = make_regression(n_features=self.n_features, n_informative=2, random_state=0, shuffle=False)
        ADBregr = RandomForestRegressor(random_state=0, n_estimators=100)
        ADBregr.fit(X, Y)
        df1 = pd.DataFrame(ADBregr.predict(self.array_XcountVectorizer))
        return df1


#---------------------------------AdaBoost ile Sınıflandırma SÜTUN ---------------------------------------------------------------------

adaboost("E").show()
adaboost("F").show()
adaboost("J").show()
adaboost("K").show()
adaboost("L").show()




#----------------------------------AdaBoost ile Regresyon----------------------------------------------------------------------------------------
carFuel=pd.DataFrame()
carFuel['yakit']=pd.DataFrame(adaboost("E").adaBoostRegresyon(3))


carGeer=pd.DataFrame()
carGeer['vites']=pd.DataFrame(adaboost("F").adaBoostRegresyon(2))

carCekis=pd.DataFrame()
carCekis['cekiş']=pd.DataFrame(adaboost("J").adaBoostRegresyon(4))

carSahip=pd.DataFrame()
carSahip['kimden']=pd.DataFrame(adaboost("K").adaBoostRegresyon(2))

carDurum=pd.DataFrame()
carDurum['durumu']=pd.DataFrame(adaboost("L").adaBoostRegresyon(3))

#--------------------------ÖLÇEKLENDİRMELER---------------------------------------------------------------------
data=pd.read_excel(path)
df=pd.DataFrame(data)

carId=df['marka']
carSeri=df['seri']
carYears=df[['yil']]
#carFuel=df['yakit']
#carGeer=df['vites']
carKm=df[['km']]
carHp=df['motor gucu']
carMv=df['motor hacmi']
#carCekis=df['cekis']
#carSahip=df['kimden']
#carDurum=df['durumu']
carModel=df['model']
carPrice=df['fiyat']

# yıl için 0-1 ölçeklendirmesi
resultYears=pd.DataFrame()
y_min_max_scalar=preprocessing.MinMaxScaler(feature_range=(0,1))
y_scaled=y_min_max_scalar.fit_transform(carYears)
resultYears['yıl']=pd.DataFrame(y_scaled)
#km için ölçeklendirme
resultKm=pd.DataFrame()
km_min_max_scalar=preprocessing.MinMaxScaler(feature_range=(0,1))
km_scaled=km_min_max_scalar.fit_transform(carKm)
resultKm['km']=pd.DataFrame(km_scaled)

#motor gücü için ölçeklendirme

resultHp=pd.DataFrame()
resultCarHp=carHp.str.extract('([0-9.\s]+)',expand=False).str.strip()
resultCarHp=resultCarHp.values.reshape(-1,1) #ValueError: Expected 2D array, got 1D array instead: hatasından kaçıyorum
Hp_min_max_scalar=preprocessing.MinMaxScaler(feature_range=(0,1))
Hp_scaled=Hp_min_max_scalar.fit_transform(resultCarHp)
resultHp['motor gücü']=pd.DataFrame(Hp_scaled)


#motor hacmi için ölçeklendirme

resultMv=pd.DataFrame()
resultCarMv=carMv.str.extract('([0-9.\s]+)',expand=False).str.strip()
resultCarMv=resultCarMv.values.reshape(-1,1) #ValueError: Expected 2D array, got 1D array instead: hatasından kaçıyorum
mv_min_max_scalar=preprocessing.MinMaxScaler(feature_range=(0,1))
mv_scaled=mv_min_max_scalar.fit_transform(resultCarMv)
resultMv['motor hacmi']=pd.DataFrame(mv_scaled)



# araç modeli ve fiyat sütunlarındaki sayısal hariç verileri kaldırdım.

resultCarModel=carModel.str.extract('([0-9.\s]+)',expand=False).str.strip()
resultCarPrice=carPrice.str.extract('([0-9.\s]+)',expand=False).str.strip()

# sonuçları result.xlsx excel dosyasına yazdırdım

resultDf=pd.concat([carId,carSeri,resultCarModel,resultYears,carFuel,carGeer,resultKm,resultHp,resultMv,carCekis,carSahip,carDurum,resultCarPrice],axis=1) #axis=1 ile verileri yan yana olacak şekilde excelle aktardım.
resultDf.to_excel('result.xlsx')
