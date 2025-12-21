import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

def show_sidebar():
    st.sidebar.markdown("## Input Features")
    mode = st.sidebar.radio("📂 Pilih metode input:", ["Upload CSV", "Manual Input"])

    if mode == "Upload CSV":
        file = st.sidebar.file_uploader("Upload file CSV")
        if file:
            df = pd.read_csv(file)
            return mode, df
        return mode, None

    # Manual Input
    if "manual_rows" not in st.session_state:
        st.session_state.manual_rows = []

    st.sidebar.markdown("---")
    st.sidebar.caption(f"📦 Samples collected: `{len(st.session_state.manual_rows)}`")
    st.sidebar.write("Isi 1 sampel lalu klik ➕ Add sample:")

    form = st.sidebar.form("manual_form", clear_on_submit=False)
    col1, col2 = form.columns(2)

    with col1:
        crim = col1.number_input("CRIM", 0.0, key="crim")
        zn = col1.number_input("ZN", 0.0, key="zn")
        indus = col1.number_input("INDUS", 0.0, key="indus")
        chas = col1.selectbox("CHAS", [0, 1], key="chas")
        nox = col1.number_input("NOX", 0.0, key="nox")
        rm = col1.number_input("RM", 0.0, key="rm")
        age = col1.number_input("AGE", 0.0, key="age")

    with col2:
        dis = col2.number_input("DIS", 0.0, key="dis")
        rad = col2.number_input("RAD", 0, key="rad")
        tax = col2.number_input("TAX", 0, key="tax")
        ptratio = col2.number_input("PTRATIO", 0.0, key="ptratio")
        b = col2.number_input("B", 0.0, key="b")
        lstat = col2.number_input("LSTAT", 0.0, key="lstat")

    submitted = form.form_submit_button("➕ Add sample")



    new_row = {
        "CRIM": crim, "ZN": zn, "INDUS": indus, "CHAS": chas, "NOX": nox,
        "RM": rm, "AGE": age, "DIS": dis, "RAD": rad, "TAX": tax,
        "PTRATIO": ptratio, "B": b, "LSTAT": lstat
    }

    if submitted:
        st.session_state.manual_rows.append(new_row)
        st.sidebar.success("Sample ditambahkan ✅")

    if st.sidebar.button("🗑️ Reset samples"):
        st.session_state.manual_rows = []
        st.sidebar.warning("Semua sample dihapus.")
        return mode, None

    if st.session_state.manual_rows:
        st.sidebar.markdown("####  Preview")
        st.sidebar.dataframe(
            pd.DataFrame(st.session_state.manual_rows).tail(3),
            use_container_width=True,
            height=140
        )

    df = pd.DataFrame(
        st.session_state.manual_rows or [new_row],
        columns=FEATURES
    )

    return mode, df


def show_result(predicted_price, input_df):
    st.markdown("## 📈 Hasil Prediksi Harga Rumah (MEDV)")

    preds = list(predicted_price)

    if len(preds) == 1:
        st.metric("💰 Estimated MEDV", f"{float(preds[0]):.2f}")
        st.markdown("### 📝 Input Data")
        st.dataframe(input_df, use_container_width=True)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Sample", len(preds))
        c2.metric("Min Prediksi", f"{min(preds):.2f}")
        c3.metric("Max Prediksi", f"{max(preds):.2f}")

        out = input_df.copy()
        out["Predicted_MEDV"] = preds
        st.markdown("### 🔍 Tabel Hasil")
        st.dataframe(out, use_container_width=True)


def show_explore(input_df: pd.DataFrame, model):
    st.subheader("📊 Eksplorasi Dataset & Prediksi")

    if input_df is None:
        st.info("Upload CSV atau isi manual dulu untuk menggunakan fitur ini.")
        return

    df = input_df.copy()

    if "MEDV" in df.columns:
        st.warning("Kolom target **MEDV** ikut terupload, akan diabaikan.")
        df = df.drop(columns=["MEDV"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing", int(df.isna().sum().sum()))

    missing_cols = [c for c in FEATURES if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in FEATURES]
    if missing_cols:
        st.error(f"⚠️ Kolom kurang: {missing_cols}")
    if extra_cols:
        st.info(f"ℹ️ Kolom tambahan (akan diabaikan): {extra_cols}")

    with st.expander("🔍 Lihat Data Input"):
        st.dataframe(df, use_container_width=True)

    st.markdown("### 📊 Batch Prediction")

    try:
        preds = model.predict(df[FEATURES])
        out = df.copy()
        out["Predicted_MEDV"] = preds
        st.dataframe(out.head(50), use_container_width=True)

        st.download_button(
            "⬇️ Download Hasil Prediksi (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

        fig = plt.figure()
        plt.hist(out["Predicted_MEDV"], bins=20, color="#00BFA6")
        plt.xlabel("Predicted MEDV")
        plt.ylabel("Count")
        plt.title("Distribusi Harga Prediksi")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Gagal melakukan prediksi batch: {e}")

    st.markdown("### Feature Explorer")
    feat = st.selectbox("Pilih fitur:", FEATURES)
    if feat in df.columns:
        fig2 = plt.figure()
        plt.hist(df[feat].dropna(), bins=20, color="orange")
        plt.xlabel(feat)
        plt.ylabel("Count")
        st.pyplot(fig2)

    st.markdown("### What-if Analysis")
    if len(df) >= 1 and all(f in df.columns for f in FEATURES):
        base = df[FEATURES].iloc[0].copy()
        what_feat = st.selectbox("Ubah fitur:", ["RM", "LSTAT", "PTRATIO", "NOX"])
        center = float(base[what_feat])
        min_v = st.number_input("Min", value=center * 0.8)
        max_v = st.number_input("Max", value=center * 1.2)
        steps = st.slider("Steps", 5, 50, 20)

        xs = np.linspace(min_v, max_v, steps)
        ys = []
        for v in xs:
            row = base.copy()
            row[what_feat] = v
            row_df = pd.DataFrame([row], columns=FEATURES)
            ys.append(float(model.predict(row_df)[0]))

        fig3 = plt.figure()
        plt.plot(xs, ys, marker="o", color="#007ACC")
        plt.xlabel(what_feat)
        plt.ylabel("Predicted MEDV")
        plt.title(f"What-if: Pengaruh perubahan {what_feat}")
        st.pyplot(fig3)
    else:
        st.info("Minimal 1 baris data dengan fitur lengkap diperlukan.")


def show_about():
    st.subheader("ℹ️ Tentang Aplikasi")
    st.markdown(
        """
        Dashboard ini dibuat untuk memprediksi harga rumah berdasarkan dataset Housing.

        **Fitur utama:**
        - Prediksi harga berdasarkan input manual / CSV
        - Analisis batch & distribusi
        - What-if simulation
        - Download hasil dalam format CSV
        """
    )
