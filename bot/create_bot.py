from aiogram import Bot
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage


# API_TOKEN = "5761106314:AAHRTn5aJwpIiswWNoRpphpuZh38GD-gsP0"
API_TOKEN = "6157408135:AAGNyYeInRXTrbGVdx_qXaiWHgDxTJP2b5w"  # мой тестовый бот
bot = Bot(token=API_TOKEN)

storage = MemoryStorage()
dispatcher = Dispatcher(bot, storage=storage)

# admin_id = 1206756552  # владелец бара
admin_id = 345705084  # kuks_51
# admin_id = 375571119  # gra4evp
# admin_id = 134566371  # gleb_kukuruz
# admin_id = 5774261029  # Rayan - ведущий
