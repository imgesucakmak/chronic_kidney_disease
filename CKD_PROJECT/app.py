import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import dice_ml
from dice_ml import Dice
import matplotlib.pyplot as plt
from pathlib import Path


st.set_page_config(page_title="CKD Karar Destek Sistemi", page_icon="🩺", layout="wide")

BASE_DIR = Path(__file__).resolve().parent


def tema_uygula():
    """Uygulama için sade, yüksek kontrastlı klinik karar destek teması."""
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 92% 3%, rgba(40, 163, 164, 0.14), transparent 22rem),
                    linear-gradient(180deg, #f7fafc 0%, #eef5f8 100%);
                color: #102a43;
            }
            .block-container { max-width: 1380px; padding-top: 2.2rem; padding-bottom: 3rem; }
            [data-testid="stSidebar"] { background: #102a43; }
            [data-testid="stSidebar"] * { color: #eef7f8; }
            [data-testid="stSidebar"] .stButton > button {
                background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.23);
                color: #ffffff; font-weight: 650;
            }
            [data-testid="stSidebar"] .stButton > button:hover {
                background: #1e8f93; border-color: #1e8f93; color: #ffffff;
            }
            .ckd-hero {
                background: linear-gradient(118deg, #102a43 0%, #174d66 58%, #1e8f93 100%);
                border-radius: 18px; padding: 1.8rem 2rem; margin-bottom: 1.25rem;
                box-shadow: 0 14px 32px rgba(16,42,67,0.18);
            }
            .ckd-hero h1 { color: #ffffff; font-size: 2rem; margin: 0.15rem 0 0.45rem; }
            .ckd-hero p { color: #dceff0; margin: 0; font-size: 1rem; }
            .ckd-eyebrow { color: #97e3dc; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.12em; font-weight: 700; }
            [data-testid="stForm"] {
                background: rgba(255,255,255,0.86); border: 1px solid #d8e6eb;
                border-radius: 16px; padding: 1.15rem 1.25rem 0.4rem;
                box-shadow: 0 8px 24px rgba(16,42,67,0.06);
            }
            div[data-testid="stMetric"] {
                background: #ffffff; border: 1px solid #d6e6e9; border-radius: 14px;
                padding: 1rem; box-shadow: 0 6px 16px rgba(16,42,67,0.06);
            }
            .stButton > button, div[data-testid="stFormSubmitButton"] > button {
                background: #0f6f73; color: #ffffff; border: 1px solid #0f6f73;
                border-radius: 9px; font-weight: 700;
            }
            .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
                background: #0a5558; border-color: #0a5558; color: #ffffff;
            }
            button[data-baseweb="tab"] { font-weight: 650; color: #426273; }
            button[data-baseweb="tab"][aria-selected="true"] { color: #0f6f73; }
            [data-testid="stDataFrame"] { border: 1px solid #d6e6e9; border-radius: 12px; overflow: hidden; }
            [data-testid="stAlert"] { border-radius: 11px; }
            .ckd-step { color: #0f6f73; font-weight: 750; font-size: 0.82rem; letter-spacing: 0.04em; }
            @media (max-width: 700px) {
                .block-container { padding: 1rem 0.8rem 2rem; }
                .ckd-hero { padding: 1.35rem; border-radius: 14px; }
                .ckd-hero h1 { font-size: 1.55rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


tema_uygula()

OKUNABILIR_ISIMLER = {
    "age": "Yaş", "bp": "Kan basıncı", "sg": "İdrar yoğunluğu",
    "al": "İdrarda albümin", "su": "İdrarda şeker", "rbc": "Kırmızı kan hücresi durumu",
    "pc": "Pus hücresi durumu", "pcc": "Pus hücre kümesi", "ba": "Bakteri",
    "bgr": "Kan şekeri", "bu": "Kan üresi", "sc": "Kreatinin", "sod": "Sodyum",
    "pot": "Potasyum", "hemo": "Hemoglobin", "pcv": "Hematokrit",
    "wbcc": "Beyaz kan hücresi sayısı", "rbcc": "Kırmızı kan hücresi sayısı",
    "htn": "Hipertansiyon", "dm": "Diyabet", "cad": "Koroner arter hastalığı",
    "appet": "İştah", "pe": "Ödem", "ane": "Anemi",
}

KISALTMALAR = {
    "bp": "Kan basıncı (mmHg).",
    "sg": "İdrar yoğunluğu; idrarın ne kadar yoğun olduğunu ifade eder.",
    "al": "İdrarda albümin düzeyi; veri setinde 0–5 olarak kodlanır.",
    "su": "İdrarda şeker düzeyi; veri setinde 0–5 olarak kodlanır.",
    "rbc": "İdrar incelemesinde kırmızı kan hücrelerinin normal/anormal durumu.",
    "pc": "İdrar incelemesinde pus hücrelerinin normal/anormal durumu.",
    "pcc": "Pus hücre kümelerinin varlığı/yokluğu.",
    "ba": "İdrar incelemesinde bakteri varlığı/yokluğu.",
    "bgr": "Kan şekeri değeri (mg/dL).",
    "bu": "Kan üresi değeri (mg/dL).",
    "sc": "Kreatinin değeri (mg/dL).",
    "sod": "Sodyum değeri (mEq/L).",
    "pot": "Potasyum değeri (mEq/L).",
    "hemo": "Hemoglobin değeri (g/dL).",
    "pcv": "Hematokrit; kanın hücrelerden oluşan hacim yüzdesi.",
    "wbcc": "Beyaz kan hücresi sayısı (hücre/mm³).",
    "rbcc": "Kırmızı kan hücresi sayısı (milyon hücre/mm³).",
    "htn": "Hipertansiyon bilgisi.", "dm": "Diyabet bilgisi.",
    "cad": "Koroner arter hastalığı bilgisi.", "appet": "İştah durumu.",
    "pe": "Ödem bilgisi.", "ane": "Anemi bilgisi.",
}


@st.cache_resource
def yukle():
    model = joblib.load(BASE_DIR / "ckd_model.pkl")
    columns = joblib.load(BASE_DIR / "columns.pkl")
    explainer = joblib.load(BASE_DIR / "shap_explainer.pkl")
    permitted_range = joblib.load(BASE_DIR / "permitted_range.pkl")
    features_to_vary = joblib.load(BASE_DIR / "features_to_vary.pkl")
    train_df = pd.read_csv(BASE_DIR / "train_data.csv").astype(float)

    continuous_features = [
        "age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv",
        "wbcc", "rbcc", "sg", "al", "su",
    ]
    data = dice_ml.Data(dataframe=train_df, continuous_features=continuous_features, outcome_name="class")

    class XGBoostCevirmen:
        def __init__(self, ana_model):
            self.model = ana_model
            if hasattr(ana_model, "classes_"):
                self.classes_ = ana_model.classes_

        def predict(self, X):
            return self.model.predict(X.astype(float))

        def predict_proba(self, X):
            return self.model.predict_proba(X.astype(float))

    dice_model = dice_ml.Model(model=XGBoostCevirmen(model), backend="sklearn")
    exp = Dice(data, dice_model, method="genetic")
    return model, columns, explainer, exp, permitted_range, features_to_vary


model, columns, explainer, exp, permitted_range, features_to_vary = yukle()

varsayilanlar = {
    "age": 50, "bp": 70, "sg": 1.020, "al": 0, "su": 0,
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "bgr": 120, "bu": 25, "sc": 0.9, "sod": 140, "pot": 4.5,
    "hemo": 15.0, "pcv": 45, "wbcc": 8000, "rbcc": 5.0,
    "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no",
}

for key, value in varsayilanlar.items():
    if key not in st.session_state:
        st.session_state[key] = value


def riskli_hasta_doldur():
    st.session_state.update({
        "age": 68, "bp": 90, "sg": 1.010, "al": 3, "su": 2, "rbc": "abnormal",
        "pc": "abnormal", "pcc": "present", "ba": "present", "bgr": 210, "bu": 80,
        "sc": 4.5, "sod": 130, "pot": 5.5, "hemo": 9.5, "pcv": 28, "wbcc": 11000,
        "rbcc": 3.2, "htn": "yes", "dm": "yes", "cad": "no", "appet": "poor",
        "pe": "yes", "ane": "yes",
    })


def saglikli_hasta_doldur():
    st.session_state.update(varsayilanlar)


def hasta_verisi_olustur(values):
    data = pd.DataFrame([{
        "age": values["age"], "bp": values["bp"], "sg": values["sg"],
        "al": values["al"], "su": values["su"],
        "rbc": 1.0 if values["rbc"] == "normal" else 0.0,
        "pc": 1.0 if values["pc"] == "normal" else 0.0,
        "pcc": 1.0 if values["pcc"] == "present" else 0.0,
        "ba": 1.0 if values["ba"] == "present" else 0.0,
        "bgr": values["bgr"], "bu": values["bu"], "sc": values["sc"],
        "sod": values["sod"], "pot": values["pot"], "hemo": values["hemo"],
        "pcv": values["pcv"], "wbcc": values["wbcc"], "rbcc": values["rbcc"],
        "htn": 1.0 if values["htn"] == "yes" else 0.0,
        "dm": 1.0 if values["dm"] == "yes" else 0.0,
        "cad": 1.0 if values["cad"] == "yes" else 0.0,
        "appet": 1.0 if values["appet"] == "good" else 0.0,
        "pe": 1.0 if values["pe"] == "yes" else 0.0,
        "ane": 1.0 if values["ane"] == "yes" else 0.0,
    }])
    return data[columns].astype(float)


def kontrol_tablosu(hasta_verisi):
    referanslar = {
        "hemo": ("Hemoglobin", 12.0, 17.0, "g/dL"),
        "sc": ("Kreatinin", 0.5, 1.2, "mg/dL"),
        "bgr": ("Kan şekeri", 70, 140, "mg/dL"),
        "bp": ("Kan basıncı", 60, 90, "mmHg"),
        "bu": ("Kan üresi", 10, 50, "mg/dL"),
    }
    rows = []
    for column, (name, minimum, maximum, unit) in referanslar.items():
        value = float(hasta_verisi[column].values[0])
        durum = "Düşük" if value < minimum else "Yüksek" if value > maximum else "Aralık içinde"
        rows.append({
            "Parametre": name,
            "Değer": f"{value:g} {unit}",
            "Gösterim aralığı": f"{minimum}–{maximum} {unit}",
            "Durum": durum,
        })
    return pd.DataFrame(rows)


def simule_et(hasta_verisi):
    cf_df, hata_mesaji, genisletilmis_kullanildi = None, None, False
    try:
        cf = exp.generate_counterfactuals(
            hasta_verisi, total_CFs=1, desired_class="opposite",
            features_to_vary=features_to_vary, permitted_range=permitted_range,
            proximity_weight=0.2, diversity_weight=5.0, verbose=False,
        )
        candidate = cf.cf_examples_list[0].final_cfs_df if cf and cf.cf_examples_list else None
        if candidate is not None and not candidate.empty:
            cf_df = candidate
    except Exception as error:
        hata_mesaji = str(error)

    if cf_df is None:
        try:
            dynamic_range = permitted_range.copy() if isinstance(permitted_range, dict) else {}
            for column in dynamic_range:
                current_value = float(hasta_verisi[column].values[0])
                dynamic_range[column] = [
                    min(dynamic_range[column][0], current_value),
                    max(dynamic_range[column][1], current_value),
                ]
            cf = exp.generate_counterfactuals(
                hasta_verisi, total_CFs=1, desired_class="opposite",
                features_to_vary=features_to_vary, permitted_range=dynamic_range,
                proximity_weight=0.2, diversity_weight=5.0, verbose=False,
            )
            candidate = cf.cf_examples_list[0].final_cfs_df if cf and cf.cf_examples_list else None
            if candidate is not None and not candidate.empty:
                cf_df, genisletilmis_kullanildi = candidate, True
        except Exception as error:
            hata_mesaji = str(error)
    return cf_df, genisletilmis_kullanildi, hata_mesaji


st.markdown(
    """
    <section class="ckd-hero">
        <div class="ckd-eyebrow">Klinik karar destek · Akademik demonstrasyon</div>
        <h1>🩺 CKD Karar Destek Sistemi</h1>
        <p>Girilen bilgilerin model çıktısını görün, ardından açıklamayı klinik bağlamla birlikte değerlendirin.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Görünüm ve örnekler")
    gorunum = st.radio("Sonuç ekranı", ["Doktor Görünümü", "Hasta Görünümü"])
    st.divider()
    st.caption("Hızlı deneme için örnek veriler")
    st.button("🔴 Riskli Hasta Yükle", on_click=riskli_hasta_doldur, width="stretch")
    st.button("🟢 Sağlıklı Hasta Yükle", on_click=saglikli_hasta_doldur, width="stretch")
    with st.expander("Kısaltmalar hakkında"):
        st.write("Her alanın yanındaki `?` simgesine gelerek kısa açıklamayı görebilirsiniz.")

st.markdown("<div class='ckd-step'>ADIM 1 / 2</div>", unsafe_allow_html=True)
st.subheader("Hasta bilgilerini girin")
st.caption("Modelin kullandığı alanlar üç anlaşılır grupta sunulur. Alan adının yanındaki `?` simgesinde kısa açıklama vardır.")

with st.form("hasta_veri_formu"):
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown("#### Genel ve idrar bulguları")
        age = st.number_input("Yaş", min_value=0, max_value=100, key="age", help="Hastanın yaşı.")
        bp = st.number_input("Kan basıncı — bp (mmHg)", min_value=40, max_value=120, key="bp", help=KISALTMALAR["bp"])
        sg = st.selectbox("İdrar yoğunluğu — sg", [1.005, 1.010, 1.015, 1.020, 1.025], key="sg", help=KISALTMALAR["sg"])
        al = st.number_input("İdrarda albümin — al (0–5)", min_value=0, max_value=5, key="al", help=KISALTMALAR["al"])
        su = st.number_input("İdrarda şeker — su (0–5)", min_value=0, max_value=5, key="su", help=KISALTMALAR["su"])
        rbc = st.selectbox("Kırmızı kan hücresi durumu — rbc", ["normal", "abnormal"], key="rbc", help=KISALTMALAR["rbc"])
        pc = st.selectbox("Pus hücresi durumu — pc", ["normal", "abnormal"], key="pc", help=KISALTMALAR["pc"])
        pcc = st.selectbox("Pus hücre kümesi — pcc", ["present", "notpresent"], key="pcc", help=KISALTMALAR["pcc"])
        ba = st.selectbox("Bakteri — ba", ["notpresent", "present"], key="ba", help=KISALTMALAR["ba"])

    with col2:
        st.markdown("#### Kan ve laboratuvar değerleri")
        bgr = st.number_input("Kan şekeri — bgr (mg/dL)", min_value=0, max_value=500, key="bgr", help=KISALTMALAR["bgr"])
        bu = st.number_input("Kan üresi — bu (mg/dL)", min_value=0, max_value=400, key="bu", help=KISALTMALAR["bu"])
        sc = st.number_input("Kreatinin — sc (mg/dL)", min_value=0.0, max_value=80.0, key="sc", help=KISALTMALAR["sc"])
        sod = st.number_input("Sodyum — sod (mEq/L)", min_value=0, max_value=200, key="sod", help=KISALTMALAR["sod"])
        pot = st.number_input("Potasyum — pot (mEq/L)", min_value=0.0, max_value=50.0, key="pot", help=KISALTMALAR["pot"])
        hemo = st.number_input("Hemoglobin — hemo (g/dL)", min_value=0.0, max_value=20.0, key="hemo", help=KISALTMALAR["hemo"])
        pcv = st.number_input("Hematokrit — pcv (%)", min_value=0, max_value=60, key="pcv", help=KISALTMALAR["pcv"])
        wbcc = st.number_input("Beyaz kan hücresi — wbcc (hücre/mm³)", min_value=0, max_value=30000, key="wbcc", help=KISALTMALAR["wbcc"])
        rbcc = st.number_input("Kırmızı kan hücresi — rbcc (milyon/mm³)", min_value=0.0, max_value=10.0, key="rbcc", help=KISALTMALAR["rbcc"])

    with col3:
        st.markdown("#### Eşlik eden durumlar")
        st.caption("Bu alanlar modelin kullandığı geçmiş/klinik durum bilgisidir.")
        htn = st.selectbox("Hipertansiyon — htn", ["no", "yes"], key="htn", help=KISALTMALAR["htn"])
        dm = st.selectbox("Diyabet — dm", ["no", "yes"], key="dm", help=KISALTMALAR["dm"])
        cad = st.selectbox("Koroner arter hastalığı — cad", ["no", "yes"], key="cad", help=KISALTMALAR["cad"])
        appet = st.selectbox("İştah — appet", ["good", "poor"], key="appet", help=KISALTMALAR["appet"])
        pe = st.selectbox("Ödem — pe", ["no", "yes"], key="pe", help=KISALTMALAR["pe"])
        ane = st.selectbox("Anemi — ane", ["no", "yes"], key="ane", help=KISALTMALAR["ane"])
        st.info("Kısaltmalar veri setiyle uyum için korunmuştur; anlaşılır alan adı önce gösterilir.")

    st.divider()
    st.caption("Analiz düğmesi, yalnızca bu formdaki değerleri kullanarak model çıktısını hesaplar.")
    submit_button = st.form_submit_button("🔍 Risk Analizi Yap", width="stretch")

if submit_button:
    values = {
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su, "rbc": rbc, "pc": pc,
        "pcc": pcc, "ba": ba, "bgr": bgr, "bu": bu, "sc": sc, "sod": sod,
        "pot": pot, "hemo": hemo, "pcv": pcv, "wbcc": wbcc, "rbcc": rbcc,
        "htn": htn, "dm": dm, "cad": cad, "appet": appet, "pe": pe, "ane": ane,
    }
    hasta_verisi = hasta_verisi_olustur(values)
    tahmin = int(model.predict(hasta_verisi)[0])
    class_1_idx = int(np.where(model.classes_ == 1)[0][0])
    olasilik = float(model.predict_proba(hasta_verisi)[0][class_1_idx])

    st.divider()
    st.markdown("<div class='ckd-step'>ADIM 2 / 2</div>", unsafe_allow_html=True)
    st.subheader("Model sonucu")

    if gorunum == "Doktor Görünümü":
        col_a, col_b, col_c = st.columns([1, 1, 1.4])
        with col_a:
            st.metric("Tahmin", "CKD sınıfı" if tahmin == 1 else "notCKD sınıfı")
        with col_b:
            st.metric("Model Skoru (CKD=1)", f"{olasilik:.1%}")
        with col_c:
            st.caption("Model skoru, modelin CKD=1 sınıfına eğilimidir; kalibre edilmiş klinik olasılık değildir.")

        if tahmin == 1:
            st.warning("Model, girilen bilgileri CKD sınıfına daha yakın buldu. Bu sonuç tek başına tanı değildir.")
        else:
            st.success("Model, girilen bilgileri notCKD sınıfına daha yakın buldu. Bu sonuç klinik değerlendirmeyi değiştirmez.")

        tab_kontrol, tab_shap, tab_simulasyon = st.tabs([
            "Kontrol edilen değerler", "SHAP açıklaması", "Model simülasyonu",
        ])

        with tab_kontrol:
            st.markdown("#### Seçili laboratuvar kontrolü")
            st.caption("Bu aralıklar yalnızca uygulama içi gösterim kontrolüdür; hasta için kesin klinik referans değildir.")
            st.dataframe(kontrol_tablosu(hasta_verisi), hide_index=True, width="stretch")

        with tab_shap:
            st.markdown("#### Modeli en çok etkileyen girdiler")
            st.caption("Grafik tek kayıt için model etkilerini gösterir; nedensel ilişki göstermez.")
            with st.expander("Grafik nasıl yorumlanmalı?", expanded=False):
                st.markdown(
                    "**Bu grafik neyi gösterir?** Her satır, o kayıt için modele girilen bir bilgiyi temsil eder. "
                    "Üstteki satırlar model sonucunu daha fazla etkileyen girdilerdir.\n\n"
                    "**Yön ne anlama gelir?** Çubukların yönü, ilgili girdinin modelin açıklanan çıktısını hangi tarafa "
                    "çektiğini gösterir. Bu etki, yalnızca modelin verdiği sonuç içindir; klinik nedensellik veya "
                    "tedavi hedefi değildir.\n\n"
                    "**Nasıl kullanılmalı?** Girdinin doğru birim ve doğru tarihle kaydedildiğini kontrol edin; ardından "
                    "grafiği klinik öykü, muayene ve güncel tetkiklerle birlikte değerlendirin. Tek bir SHAP satırından "
                    "tanı veya müdahale sonucu çıkarılmamalıdır."
                )
            shap_values = explainer.shap_values(hasta_verisi)
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=hasta_verisi.iloc[0].values,
                feature_names=[OKUNABILIR_ISIMLER.get(column, column) for column in hasta_verisi.columns],
            )
            fig, _ = plt.subplots(figsize=(9, 6))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with tab_simulasyon:
            if tahmin != 1:
                st.info("Model kayıt için zaten notCKD sınıfına yakın sonuç verdiğinden ek simülasyon gösterilmiyor.")
            else:
                st.markdown("#### Hipotetik gözlem profili")
                st.info("Bu bölüm tedavi planı değildir. Yalnızca modelin varsayımsal bir alternatif profilde farklı sınıfa geçip geçmediğini gösterir.")
                with st.spinner("Model simülasyonu hesaplanıyor..."):
                    cf_df, genisletilmis_kullanildi, hata_mesaji = simule_et(hasta_verisi)

                if cf_df is None:
                    st.warning("Abstain (geri çekilme): Model bu kayıt için geçerli bir simülasyon üretemedi.")
                    if hata_mesaji:
                        st.caption(f"Teknik detay: {hata_mesaji[:200]}")
                else:
                    if genisletilmis_kullanildi:
                        st.caption("ℹ️ Sıkı aralıkta çözüm bulunamadığı için daha az kısıtlı bir arama kullanıldı.")
                    rows = []
                    for column in hasta_verisi.columns:
                        old_value, new_value = hasta_verisi[column].values[0], cf_df[column].values[0]
                        if str(old_value) != str(new_value):
                            try:
                                difference = f"{float(new_value) - float(old_value):+.1f}"
                            except (ValueError, TypeError):
                                difference = "Durum değişikliği"
                            rows.append({
                                "Özellik": OKUNABILIR_ISIMLER.get(column, column),
                                "Mevcut değer": old_value,
                                "Modelin alternatif değeri": new_value,
                                "Fark": difference,
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                    else:
                        st.warning("Model karar sınırını değiştirecek bir profil bulunamadı.")

    else:
        if tahmin == 1:
            st.error("Model, girilen bilgileri CKD sınıfına daha yakın buldu.")
        else:
            st.success("Model, girilen bilgileri notCKD sınıfına daha yakın buldu.")
        st.metric("Model Skoru (CKD=1)", f"{olasilik:.0%}")
        st.caption("Bu skor tanı değildir ve gerçek hayattaki hastalık olasılığı olarak yorumlanmamalıdır.")
        with st.expander("Model açıklaması (SHAP) hakkında", expanded=False):
            st.markdown(
                "Doktor görünümünde bulunan SHAP grafiği, modelin bu sonucu hangi girilen bilgilerden etkilenerek "
                "oluşturduğunu açıklar. Grafik, bir bulgunun hastalığa neden olduğunu veya o bulgunun nasıl "
                "değiştirileceğini söylemez. Bu nedenle grafik; tanı, tedavi veya yaşam tarzı kararı için tek başına "
                "kullanılmamalı, sağlık profesyoneliyle birlikte yorumlanmalıdır."
            )
        st.warning("Sonucunuzu tıbbi öykünüz ve güncel tetkiklerinizle birlikte bir sağlık profesyoneli değerlendirmelidir.")

    st.divider()
    st.warning("⚠️ **Kullanım uyarısı:** Bu sistem akademik amaçlı bir yapay zekâ demonstrasyonudur; tanı, tıbbi tavsiye veya tedavi önerisi vermez.")
