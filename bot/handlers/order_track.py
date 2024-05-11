from aiogram import types
from aiogram.dispatcher import Dispatcher, FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.dispatcher.filters import Text

from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.utils.markdown import hlink
from bot.unique_links_parse import get_unique_links, load_links_by_user_id
from bot.sqlalchemy_orm import session, VisitorPerformance, Recommendations
from bot.create_bot import admin_id
import random
import os
from bot.utils.download_youtube_track import YouTubeTrackDownloader

print(f"Текущий рабочий каталог: {os.getcwd()}")


class FSMOrderTrack(StatesGroup):
    track_url = State()


async def get_next_link_round_command(message: types.Message):
    counter_empty = 0
    for user_id, value in user_ids.items():
        list_links, username = value
        if len(list_links):
            await message.answer(f"{hlink('Track', list_links.pop(0))} ordered by @{username}", parse_mode='HTML')
        else:
            counter_empty += 1

    if counter_empty == len(user_ids):
        await message.answer('Oops, Seems the songs are over.')


async def order_track_command(message: types.Message):
    await message.answer('Good! Add youtube link 😉')
    await FSMOrderTrack.track_url.set()


async def state_order_track_is_invalid(message: types.Message):
    await message.answer('You added the link incorrectly, please try again 😉')


async def add_link(message: types.Message, state: FSMContext):
    await state.finish()

    user_id = message.from_user.id

    if user_id not in user_ids:
        user_ids[user_id] = ([], message.from_user.username)

    user_ids[user_id][0].append(message.text)
    print(message.text)
    performance = VisitorPerformance(user_id=user_id, url=message.text, created_at=message.date)
    session.add(performance)
    session.commit()

    await message.answer('Success! Sing better than the original, I believe in you 😇')
    await download_track(message)
    # await get_recommendation(message)


async def get_recommendation(message: types.Message):
    user_id = message.from_user.id

    links = links_by_user_id.get(str(user_id))
    if links:
        link = links.pop(random.randint(0, len(links) - 1))
        type_link = 'user_link'
    else:
        link = random.choice(unique_links)
        type_link = 'random_link'

    rec_message = await message.answer(f"{link}\n\nTest recommendation", parse_mode='HTML')

    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton(text="Order this track", callback_data=f'order_this_track'))
    await rec_message.edit_reply_markup(keyboard)

    recommendation = Recommendations(user_id=user_id, message_id=rec_message.message_id, url=link, rec_type=type_link,
                                     is_accepted=False, created_at=message.date, updated_at=message.date)
    session.add(recommendation)
    session.commit()


async def download_track(message: types.Message):
    downloader = YouTubeTrackDownloader(url=message.text)
    filename = downloader.download()
    print(filename)
    await message.answer("Ссылка была сохранена")


async def callback_order_this_track(callback: types.CallbackQuery):
    recommendation = session.query(Recommendations).filter(Recommendations.message_id == callback.message.message_id).first()
    # Проверить когда может придти None?
    recommendation.is_accepted = True
    recommendation.updated_at = callback.message.date

    text = callback.message.text.replace('\n\nTest recommendation', '')
    performance = VisitorPerformance(user_id=callback.from_user.id,
                                     url=text,
                                     created_at=callback.message.date)
    session.add(performance)

    if callback.from_user.id not in user_ids:
        user_ids[callback.from_user.id] = ([], callback.from_user.username)

    user_ids[callback.from_user.id][0].append(recommendation.url)

    await callback.answer('Success! Sing better than the original, I believe in you 😇')
    await callback.message.edit_text(f"✅ {hlink('Track', recommendation.url)} is ordered", parse_mode='HTML')
    session.commit()


def register_handlers(dispatcher: Dispatcher):
    dispatcher.register_message_handler(order_track_command, commands=['order_track'])
    dispatcher.register_message_handler(order_track_command, Text('Order a track', ignore_case=True))

    dispatcher.register_message_handler(
        get_next_link_round_command,
        lambda message: message.from_user.id == admin_id,
        commands=['get_next_link_round']
    )
    dispatcher.register_message_handler(
        get_next_link_round_command,
        lambda message: message.from_user.id == admin_id,
        Text('Get next link round')
    )

    dispatcher.register_message_handler(
        add_link,
        Text(startswith=[
            'https://www.youtube.com/watch?v=',
            'https://youtu.be/',
            'https://xminus.me/track/',
            'https://x-minus.cc/track/',
            'https://x-minus.me/track/',
            'https://x-minus.pro/track/',
            'https://x-minus.club/track/',
            'https://xm-rus.top/track/']
        ),
        state=FSMOrderTrack.track_url
    )
    dispatcher.register_message_handler(
        state_order_track_is_invalid,
        content_types='any',
        state=FSMOrderTrack.track_url
    )

    dispatcher.register_callback_query_handler(
        callback_order_this_track,
        Text(startswith='order_this_track')
    )


user_ids = {}

unique_links = get_unique_links('id_url_all.csv')
links_by_user_id = load_links_by_user_id('links_by_user_id.json')
