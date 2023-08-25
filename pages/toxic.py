import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    st.title("Оценка токсичности сообщений")

    # Загрузка модели
    model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()
        
    def text2toxicity(text, aggregate=True):
        """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
            proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
        if isinstance(text, str):
            proba = proba[0]
        if aggregate:
            return 1 - proba.T[0] * (1 - proba.T[-1])
        return proba

    message = st.text_area("Введите сообщение для оценки:")
    if st.button("Оценить"):
        if message:
            toxicity_score = text2toxicity(message)
            st.write(f"Степень токсичности: {toxicity_score:.4f}")


    st.write("### Если вы хотите воспользоваться Telegram ботом для этой задачи, вы можете найти его здесь:")
    st.write("[Ссылка на Telegram бота](https://t.me/ToxicElbBot)")

st.sidebar.image('images/toxic.jpeg', use_column_width=True) 



if __name__ == "__main__":
    main()
