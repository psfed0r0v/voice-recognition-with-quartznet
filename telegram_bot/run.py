from pathlib import Path

import requests
from aiogram.types import ContentType, File, Message

import logging

from aiogram import Bot, Dispatcher, executor, types

API_TOKEN = 'YOUR_TOKEN_HERE'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'restart'])
async def send_welcome(message: types.Message):
    m = '''Hi!\nI'm Voice Recognition Bot!\nSend voice message and I'll send recognized text.\n(I only understand english language)'''
    await message.reply(m)


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer('Sorry, I understand only voice messages, try again')


async def handle_file(file: File, file_name: str, path: str):
    Path(f"{path}").mkdir(parents=True, exist_ok=True)

    await bot.download_file(file_path=file.file_path, destination=f"{path}/{file_name}")

    files = [('file', open(f"{path}/{file_name}", 'rb'))]
    response = requests.post(url='http://127.0.0.1:80/upload-voice', files=files)
    Path(f"{path}/{file_name}").unlink()
    print(response.json())

    return response


@dp.message_handler(content_types=[ContentType.VOICE])
async def voice_message_handler(message: Message):
    voice = await message.voice.get_file()
    path = "../telegram_bot/voices"

    response = await handle_file(file=voice, file_name=f"{voice.file_id}.ogg", path=path)
    m = response.json().get('text', 'Something went wrong, try again')
    if not len(m):
        m = "I can't recognize anything, say louder please"
    await message.answer(m)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
