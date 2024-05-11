from aiogram import types
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.dispatcher.filters import Text
from karaoke_bot.bot_old_version.create_bot import bot, admin_id


async def start(message: types.Message):
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)

    if message.from_user.id != admin_id:
        keyboard.add(KeyboardButton("Order a track"))
        keyboard.add(KeyboardButton("Join a group of karaoke lovers"))
    else:
        keyboard.add(KeyboardButton("Get next link round"))

    bot_info = await bot.get_me()
    await message.answer(
        f"Welcome, {message.from_user.first_name}!\nI'm  - <b>{bot_info.first_name}</b>, "
        f"the telegram bot of my favorite Venue bar. "
        f"And I'm here to help you sing as many songs as possible! "
        f"I hope you warmed up your vocal cords. 😏",
        parse_mode='html',
        reply_markup=keyboard
    )


async def join_a_group(message: types.Message):
    await message.answer('https://t.me/+JPb01AZkQgxkOGMy')


async def send_hello_text(message: types.Message):
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)

    if message.from_user.id != admin_id:
        keyboard.add(KeyboardButton("Order a track"))
        keyboard.add(KeyboardButton("Join a group of karaoke lovers"))
    else:
        keyboard.add(KeyboardButton("Get next link round"))
    await message.answer('Hello', reply_markup=keyboard)


async def cancel_command(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Ok")
        return None
    await state.finish()
    await message.reply("Ok")


def register_handlers(dispatcher: Dispatcher):

    dispatcher.register_message_handler(cancel_command, Text(equals='cancel', ignore_case=True), state='*')
    dispatcher.register_message_handler(cancel_command, commands=['cancel'], state='*')

    dispatcher.register_message_handler(start, commands=['start'], state='*')
    dispatcher.register_message_handler(
        join_a_group,
        Text(equals='Join a group of karaoke lovers', ignore_case=True)
    )

    dispatcher.register_message_handler(
        send_hello_text,
        lambda message: message.text not in ['Order a track', 'order a track'],
        content_types='any'
    )
