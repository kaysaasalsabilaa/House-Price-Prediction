import streamlit as st
from model import load_model, predict_price
from ui import show_sidebar, show_result, show_explore, show_about


st.set_page_config(page_title="🏡 House Price Estimator", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-image:
            linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)),
            url("https://images.unsplash.com/photo-1600585154340-be6161a56a0c");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center; padding: 2rem 0; background-color: #f4f6f7; border-radius: 10px;'>
        <h1 style='color: #00BFA6;'>🏡 House Price Estimator</h1>
        <p style='font-size: 1.2rem; color: #333;'>A machine learning dashboard for predicting housing prices using real-world features.</p>
    </div>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def get_model():
    return load_model()

model = get_model()


mode, input_df = show_sidebar()


tabs = st.tabs(["🔮 Predict", "📊 Explore", "ℹ️ About"])

with tabs[0]:
    if input_df is not None:
        pred = predict_price(model, input_df)
        show_result(pred, input_df)
    else:
        st.info("Isi input dulu ya (manual/CSV).")

with tabs[1]:
    show_explore(input_df, model)

with tabs[2]:
    show_about()

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with Streamlit | Designed by Kaysa</p>",
    unsafe_allow_html=True
)
