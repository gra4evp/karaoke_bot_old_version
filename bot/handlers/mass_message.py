from aiogram import types
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
import time
import csv
from karaoke_bot.bot_old_version.sqlalchemy_orm import session, VisitorPerformance
from karaoke_bot.bot_old_version.create_bot import bot


class FSMMassMessage(StatesGroup):
    text = State()
    image = State()
    confirm = State()


async def mass_message(message: types.Message):
    await message.answer("<b>Mass Message Constructor</b>", parse_mode='HTML')

    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton(text='Skip', callback_data='mass_message_skip text'))
    await message.answer("Please enter the 💬 <b>TEXT</b> (if any) for the message",
                         reply_markup=keyboard,
                         parse_mode='HTML')
    await FSMMassMessage.text.set()


async def mass_message_text_registration(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['mass_message_text'] = message.html_text

    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton(text='Skip', callback_data='mass_message_skip image'))
    await message.answer("Great!")
    await message.answer("Now please upload the 🖼 <b>IMAGE</b> (if any) for the message",
                         reply_markup=keyboard,
                         parse_mode='HTML')
    await FSMMassMessage.image.set()


async def mass_message_image_registration(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['mass_message_image_id'] = message.photo[0].file_id
    await mass_message_confirm(message.from_user.id, state)


async def state_mass_message_image_is_invalid(message: types.Message):
    await message.reply("It seems you sent something wrong\nPlease send a <b>IMAGE</b> to the mass message\n\n",
                        parse_mode='HTML')


async def mass_message_confirm(user_id, state: FSMContext):

    confirm_text = "<b>CONFIRM SENDING</b>"
    async with state.proxy() as data:
        text = data.get('mass_message_text')
        text = confirm_text if text is None else f'{confirm_text}\n\nMESSAGE TEXT:\n{text}'
        image_id = data.get('mass_message_image_id')

    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton('✅ Confirm and Send', callback_data='mass_message send'))
    keyboard.insert(InlineKeyboardButton('✏️ Edit', callback_data='mass_message edit'))
    keyboard.add(InlineKeyboardButton('❌ Delete', callback_data='mass_message delete'))

    if image_id is not None:
        await bot.send_photo(chat_id=user_id, photo=image_id, caption=text, reply_markup=keyboard, parse_mode='HTML')
    else:
        await bot.send_message(chat_id=user_id, text=text, reply_markup=keyboard, parse_mode='HTML')

    await FSMMassMessage.confirm.set()


async def callback_mass_message_confirm(callback: types.CallbackQuery, state: FSMContext):
    action = callback.data.split(' ')[-1]
    await callback.answer()

    keyboard = InlineKeyboardMarkup()
    if action == 'send':
        keyboard.add(InlineKeyboardButton('✅ Send', callback_data='mass_message force_send'),
                     InlineKeyboardButton('<< Back', callback_data='mass_message back'))
        await callback.message.edit_reply_markup(keyboard)

    elif action == 'force_send':
        await callback.answer('✅ Mass message sent successfully!', show_alert=True)
        await callback.message.delete()

        async with state.proxy() as data:
            text = data.get('mass_message_text')
            image_id = data.get('mass_message_image_id')

        await state.finish()
        await send_mass_message(sender_id=callback.from_user.id, text=text, image_id=image_id)

    elif action == 'edit':
        keyboard.add(InlineKeyboardButton('💬 Edit text', callback_data='mass_message edit_text'))
        keyboard.insert(InlineKeyboardButton('🖼 Edit image', callback_data='mass_message edit_image'))
        keyboard.add(InlineKeyboardButton('<< Back', callback_data='mass_message back'))
        await callback.message.edit_reply_markup(keyboard)

    elif action == 'delete':
        keyboard.add(InlineKeyboardButton('❌ Delete', callback_data='mass_message force_delete'),
                     InlineKeyboardButton('<< Back', callback_data='mass_message back'))
        await callback.message.edit_reply_markup(keyboard)

    elif action == 'force_delete':
        await callback.message.answer('❌ Mass message deleted')
        await callback.message.delete()
        await state.finish()

    elif action == 'back':
        keyboard.add(InlineKeyboardButton('✅ Confirm and Send', callback_data='mass_message send'))
        keyboard.insert(InlineKeyboardButton('✏️ Edit', callback_data='mass_message edit'))
        keyboard.add(InlineKeyboardButton('❌ Delete', callback_data='mass_message delete'))
        await callback.message.edit_reply_markup(keyboard)


async def callback_mass_message_skip(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()

    skiptype = callback.data.split(' ')[-1]
    if skiptype == 'text':
        await callback.message.answer("Please upload the 🖼 <b>IMAGE</b> for the message", parse_mode='HTML')
        await FSMMassMessage.image.set()
    if skiptype == 'image':
        await FSMMassMessage.confirm.set()
        await mass_message_confirm(callback.from_user.id, state)


async def send_mass_message(sender_id: int, text: str, image_id: str):

    with open('../id_url_all.csv', encoding='u8') as fi:
        unique_user_ids = set(int(row['user_id']) for row in csv.DictReader(fi))

    user_ids = session.query(VisitorPerformance.user_id.distinct()).all()

    for user_id, in user_ids:  # type(user_id) - <tuple[int]>
        unique_user_ids.add(user_id)

    count_sended = 0
    for user_id in unique_user_ids:
        try:
            if text is not None:
                if image_id is not None:
                    await bot.send_photo(chat_id=user_id, photo=image_id, caption=text, parse_mode='HTML')
                else:
                    await bot.send_message(chat_id=user_id, text=text, parse_mode='HTML')
            else:
                await bot.send_photo(chat_id=user_id, photo=image_id, parse_mode='HTML')
            print(f'Сообщение доставлено {user_id}')
            count_sended += 1
            time.sleep(0.1)  # Можно отправлять 30 сообщений в секунду

        except Exception as e:
            print(f'Возникла ошибка при отправлении - {user_id}: {str(e)}')

    await bot.send_message(chat_id=sender_id,
                           text=f'The message was sent to {count_sended}/{len(unique_user_ids)} users')


def register_handlers(dispatcher: Dispatcher):
    dispatcher.register_message_handler(
        mass_message,
        lambda message: message.from_user.id in [345705084, 375571119, 134566371],
        commands=['mass_message']
    )

    dispatcher.register_message_handler(
        mass_message_text_registration,
        content_types=['text'],
        state=FSMMassMessage.text
    )

    dispatcher.register_message_handler(
        mass_message_image_registration,
        content_types=['photo'],
        state=FSMMassMessage.image
    )
    dispatcher.register_message_handler(
        state_mass_message_image_is_invalid,
        content_types='any',
        state=FSMMassMessage.image
    )

    dispatcher.register_callback_query_handler(
        callback_mass_message_confirm,
        Text(equals=[
            'mass_message send',
            'mass_message force_send',
            'mass_message edit',
            'mass_message delete',
            'mass_message force_delete',
            'mass_message back']
        ),
        state=FSMMassMessage.confirm
    )

    dispatcher.register_callback_query_handler(
        callback_mass_message_skip,
        Text(equals=['mass_message_skip text', 'mass_message_skip image']),
        state=[FSMMassMessage.text, FSMMassMessage.image]
    )
