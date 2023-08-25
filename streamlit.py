import streamlit as st
import ssl

# Отключение проверки SSL-сертификата
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(
    page_title='Проект. Обработка естественного языка',
    layout='wide'
)

st.sidebar.header("Home page")
c1, c2 = st.columns(2)
c2.image('images/image.jpeg')
c1.markdown("""
# Проект. Обработка естественного языка
Cостоит из 3 частей:
### 1. Классификация отзыва на поликлиники
### 2. Генерация текста GPT-моделью в стиле А.С. Пушкина, В.В. Маяковского. 
### Бонус - Кодекс Братана
### 3. Оценка степени токсичности пользовательского сообщения
""")
