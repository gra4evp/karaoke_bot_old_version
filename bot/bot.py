from aiogram.utils import executor
from create_bot import dispatcher
import handlers

handlers.common.register_handlers(dispatcher)
handlers.order_track.register_handlers(dispatcher)
handlers.mass_message.register_handlers(dispatcher)

executor.start_polling(dispatcher, skip_updates=True)
