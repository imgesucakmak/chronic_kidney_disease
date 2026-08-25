# Açıklanabilir Klinik Karar Destek Sistemi (CKD Risk Analizi)

Kronik böbrek hastalığı (Chronic Kidney Disease – CKD) risk tahmini yapan, tahminlerini
**SHAP** ve **karşı-olgusal (counterfactual) açıklamalar** ile destekleyen, doktor ve hasta
için ayrı görünümler sunan bir klinik karar destek sistemi prototipi.

Bu proje bir bitirme projesi kapsamında, akademik/demonstrasyon amacıyla geliştirilmiştir.
**Tanı, tedavi önerisi veya tıbbi tavsiye niteliği taşımaz.**

## 🔗 Canlı Demo

[Uygulamayı deneyin](#) <!-- Streamlit Cloud linkini buraya ekleyin -->

## 🎯 Projenin Amacı

Klinik karar destek sistemlerinde kullanılan açıklanabilir yapay zekâ (XAI) yöntemleri
genellikle iki sınırlılık taşır:

- **Tek tip açıklama:** Aynı çıktı hem klinisyene hem hastaya, aynı teknik detayla sunulur.
- **Eyleme geçirilemezlik:** SHAP gibi yöntemler "hangi özellik önemli" der, ama "ne
  değişirse sonuç değişir" sorusuna doğrudan cevap vermez.

Bu proje, standart SHAP analizinin üzerine **kısıtlı ve gerçekçi karşı-olgusal
açıklamalar** (DiCE) ekleyerek ve **kullanıcı tipine göre farklılaşan arayüzler**
(doktor / hasta) sunarak bu iki sınırlılığı ele almayı amaçlar.

## 🧠 Yöntem Özeti

| Aşama | Açıklama |
|---|---|
| Veri seti | UCI Chronic Kidney Disease (400 hasta, 24 öznitelik) |
| Model | XGBoost (SHAP/DiCE performansı ve klinik dengeler gözetilerek Random Forest ile karşılaştırılıp seçildi) |
| Açıklanabilirlik | SHAP (özellik katkı analizi) + DiCE (karşı-olgusal senaryo üretimi) |
| Kısıtlar | Yaş, hipertansiyon, diyabet ve koroner arter hastalığı gibi **geçmiş/değiştirilemez** özellikler sabit tutulur; laboratuvar değerleri yalnızca **klinik olarak gerçekçi aralıklarda** değiştirilebilir |
| Arayüz | Streamlit — Doktor Görünümü (teknik detay) ve Hasta Görünümü (sade dil) |

### Neden Karşı-Olgusal Açıklamalarda Kısıt Gerekli?

Kısıtlanmamış bir karşı-olgusal arama, modelin **geçmiş tanıları** (örn. hipertansiyon,
diyabet) "geri alarak" hastayı sağlıklı sınıfına geçirmeye çalışabilir — bu klinik olarak
anlamsızdır. Bu projede bu sorun tespit edilmiş ve **değiştirilemez özellik listesi** ile
**gerçekçi değer aralıkları** (`permitted_range`) tanımlanarak giderilmiştir. Standart
kısıtlarla çözüm bulunamayan nadir vakalarda, sistem şeffaf bir şekilde belirtilerek daha
geniş bir arama uzayına geçer (bkz. `app.py` içindeki `simule_et` fonksiyonu).

## 📁 Depo Yapısı

```
.
├── app.py                      # Streamlit arayüzü (doktor/hasta görünümleri)
├── chronic_kidney_disease.py   # Veri işleme, model eğitimi, SHAP/DiCE kurulumu
├── requirements.txt            # Python bağımlılıkları
├── ckd_model.pkl                # Eğitilmiş XGBoost modeli
├── shap_explainer.pkl           # SHAP TreeExplainer nesnesi
├── columns.pkl                  # Model giriş sütun sırası
├── permitted_range.pkl          # DiCE için gerçekçi değer aralıkları
├── features_to_vary.pkl         # DiCE için değiştirilebilir özellik listesi
├── train_data.csv               # Eğitim verisi (DiCE referans verisi olarak kullanılır)
└── test_data.csv                # Test verisi
```

## 🚀 Yerel Kurulum

```bash
git clone <repo-url>
cd <repo-klasoru>
pip install -r requirements.txt
streamlit run app.py
```

Uygulama açıldığında tarayıcınızda `http://localhost:8501` adresine yönlendirilirsiniz.

## 🖥️ Kullanım

1. Sol menüden **Doktor Görünümü** veya **Hasta Görünümü** seçin.
2. Sağ taraftaki "Hızlı Simülasyon" butonlarıyla örnek bir riskli/sağlıklı hasta
   yükleyebilir, ya da formu manuel doldurabilirsiniz.
3. **Risk Analizi Yap** butonuna basın.
4. Doktor görünümünde: model skoru, klinik referans aralığı kontrolü, SHAP grafiği ve
   (riskli tahminlerde) hipotetik gözlem profili sekmeler halinde sunulur.
5. Hasta görünümünde: sonuç sade bir dille özetlenir.

## ⚠️ Sınırlılıklar

- Veri seti görece küçüktür (400 kayıt); sonuçların daha büyük ve heterojen klinik veri
  setlerinde doğrulanması gerekir.
- Karşı-olgusal açıklama üretimi tüm vakalarda garanti değildir; bazı uç/karma
  vakalarda gerçekçi kısıtlar altında çözüm bulunamayabilir.
- Model çıktısı kalibre edilmiş bir klinik olasılık değildir.
- Sistem bir tanı veya tedavi aracı değildir; yalnızca akademik/demonstrasyon amaçlıdır.

## 📚 Kaynakça

Proje, aşağıdaki çalışmaların işaret ettiği literatür boşluklarından (SHAP'ın tek
başına yetersizliği, kullanıcı tipine göre farklılaşmayan açıklamalar, hasta
perspektifinin eksikliği) hareketle tasarlanmıştır:

- Salimparsa et al. (2025), Informatics
- Gambetti et al. (2025), arXiv
- Abbas, Jeong & Lee (2025), Healthcare
- Amann et al. (2022), PLOS Digital Health
- Zhang et al. (2025), Frontiers in Nutrition

## 👤 Yazar

Imge Su — Bilgisayar Mühendisliği.
