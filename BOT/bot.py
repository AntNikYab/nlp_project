import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

API_TOKEN = os.getenv('TOKEN', 'Ключа нет')

# Инициализация бота и диспетчера
logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
middleware = LoggingMiddleware()

# Инициализация модели и токенизатора
model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

# Функция для оценки токсичности текста
def text2toxicity(text, aggregate=True):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

@dp.message_handler(commands=['start'])
async def on_start(message: types.Message):
    await message.reply("Привет! Я бот для оценки токсичности сообщений. Пожалуйста, отправь мне сообщение для оценки.")

@dp.message_handler()
async def evaluate_toxicity(message: types.Message):
    text = message.text
    toxicity_score = text2toxicity(text)
    response = f"Степень токсичности: {toxicity_score:.4f}"
    await message.reply(response)

if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)