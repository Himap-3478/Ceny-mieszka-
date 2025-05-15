import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# Wczytaj model
model = load_model("model_ceny_mieszkan")

# Tytuł aplikacji
st.title("🏠 Przewidywanie cen mieszkań w Trójmiastach")

# Opis
st.markdown("""
Aplikacja przewiduje **cenę transakcyjną mieszkania z rynku wtórnego** w wybranych miejscowościach Trójmiasta małego i dużego **od 2025 do 2040 roku**.
""")

# Lista miast
miasta_duze = ["Gdańsk", "Gdynia", "Sopot"]
miasta_male = ["Wejherowo", "Reda", "Rumia"]
wszystkie_miasta = miasta_duze + miasta_male

# Wybór miasta
miasto = st.selectbox("Wybierz miejscowość:", wszystkie_miasta)

# Metraż mieszkania
metraz = st.number_input("Podaj metraż mieszkania (m²):", min_value=10, max_value=200, step=1)

# Wybór roku i kwartału
rok = st.selectbox("Wybierz rok:", list(range(2025, 2040)))
kwartal = st.selectbox("Wybierz kwartał:", [1, 2, 3, 4])

# Przycisk przewidywania
if st.button("🔍 Przewiduj cenę"):
    # Przygotuj dane wejściowe
    input_df = pd.DataFrame({
        'Miasto': [miasto],
        'Metraż': [metraz],
        'Rok': [rok],
        'Kwartał': [kwartal]
    })

    # Przewidywanie ceny
    wynik = predict_model(model, data=input_df)
    cena_pred = wynik['prediction_label'].iloc[0]

    st.success(f"💰 Przewidywana cena mieszkania to: **{round(cena_pred * metraz):,} zł** ({round(cena_pred, 2)} zł/m²)")

# Stopka
st.markdown("---")
st.caption("Model oparty na danych kwartalnych 2020–2024. Tylko rynek wtórny.")
