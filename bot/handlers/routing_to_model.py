import aiohttp
from aiogram import types
from aiogram.dispatcher import Dispatcher


FLASK_APP_URL = 'http://model:5002'  # URL Flask-приложения в Docker-сети


async def fetch_status(session, task_id):
    async with session.get(f"{FLASK_APP_URL}/status/{task_id}") as response:
        return await response.json()


async def fetch_result(session, task_id):
    async with session.get(f"{FLASK_APP_URL}/result/{task_id}") as response:
        return await response.json()


async def submit_task(message: types.Message):
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.post(f"{FLASK_APP_URL}/submit") as response:
            data = await response.json()
            task_id = data.get('task_id')
            await message.reply(f"Task submitted with ID: {task_id}")


async def check_status(message: types.Message):
    task_id = int(message.get_args())
    async with aiohttp.ClientSession(trust_env=True) as session:
        status = await fetch_status(session, task_id)
        await message.reply(f"Status for task {task_id}: {status['status']}")


async def get_result(message: types.Message):
    task_id = int(message.get_args())
    async with aiohttp.ClientSession(trust_env=True) as session:
        result = await fetch_result(session, task_id)
        await message.reply(f"Result for task {task_id}: {result['result']}")


def register_handlers(dp: Dispatcher):
    dp.register_message_handler(check_status, commands=['check_status'])
    dp.register_message_handler(submit_task, commands=['submit_task'])
    dp.register_message_handler(get_result, commands=['get_result'])
