import os
import sys
# Получаем абсолютный путь к корневой директории проекта
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Добавляем корневую директорию в путь поиска модулей Python
sys.path.append(root_dir)


from aiogram.utils import executor
from create_bot import dispatcher
import handlers

handlers.common.register_handlers(dispatcher)
handlers.order_track.register_handlers(dispatcher)
handlers.mass_message.register_handlers(dispatcher)

executor.start_polling(dispatcher, skip_updates=True)
