import streamlit as st
import textwrap
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = torch.device("cpu")
# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
model_finetuned = GPT2LMHeadModel.from_pretrained(
    'sberbank-ai/rugpt3small_based_on_gpt2',
    output_attentions = False,
    output_hidden_states = False,
)
if torch.cuda.is_available():
    model_finetuned.load_state_dict(torch.load('models/model_pushkin.pt'))
else:
    model_finetuned.load_state_dict(torch.load('models/model_pushkin.pt', map_location=torch.device('cpu')))
model_finetuned.eval()

# Function to generate text
def generate_text(prompt, temperature, top_p, max_length, top_k):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        out = model_finetuned.generate(
            input_ids,
            do_sample=True,
            num_beams=5,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            top_k=top_k,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
        )
        
    generated_text = list(map(tokenizer.decode, out))
    return generated_text

# Streamlit app
def main():
    st.title("Генерация текста GPT-моделью в стиле А.С. Пушкина")
    
    # User inputs
    prompt = st.text_area("Введите начало текста")
    temperature = st.slider("Temperature", min_value=0.2, max_value=2.5, value=1.8, step=0.1)
    top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
    max_length = st.slider("Max Length", min_value=10, max_value=300, value=100, step=10)
    top_k = st.slider("Top-k", min_value=1, max_value=500, value=500, step=10)
    num_return_sequences = st.slider("Number of Sequences", min_value=1, max_value=5, value=1, step=1)

    if st.button("Generate Text"):
        st.subheader("Generated Text:")
        for i in range(num_return_sequences):
            generated_text = generate_text(prompt, temperature, top_p, max_length, top_k)
            st.write(f"Generated Text {i + 1}:")
            wrapped_text = textwrap.fill(generated_text[0], width=80)
            st.write(wrapped_text)
            st.write("------------------")

st.sidebar.image('images/pushkin.jpeg', use_column_width=True) 

if __name__ == "__main__":
    main()