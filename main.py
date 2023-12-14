import telebot
from telebot import types
from config import *
import numpy as np
from model import *
from keras.preprocessing.sequence import pad_sequences

bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=["start"])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Analysis")
    markup.add(btn1)
    bot.send_message(message.chat.id,
                     text='Hi! My name is My Lubasha. I`m a neural network for sentiment analysis. Push the "Analysis"',
                     reply_markup=markup)


@bot.message_handler(content_types=['text'])
def analysis(message):
    if message.text == "Analysis":
        if message.from_user.username == 'alfa_timoha':
            bot.send_message(message.chat.id, text='Спасибо тебе большое, братишка')
        bot.send_message(message.chat.id, text='Please print your text')

    else:
        msg = tokenizer.texts_to_sequences(message.text)
        msg = pad_sequences(msg, maxlen=47, dtype='int32')
        sentiment = model.predict(msg, batch_size=1, verbose='auto')
        if np.argmax(sentiment) > 5:
            answer = 'negative'
        elif np.argmax(sentiment) < 2:
            answer = 'positive'
        else:
            answer = 'neutral'
        bot.send_message(message.chat.id, answer)
        bot.send_message(message.chat.id, text='Please print your text')


bot.polling(none_stop=True)