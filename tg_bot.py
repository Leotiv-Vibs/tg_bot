from io import BytesIO
import logging
from datetime import datetime

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from telegram.ext import *

DEVICE = 'cpu'
THICKNESS = 2
FONT_SCALE = 1
COLOR = (255, 0, 0)
LIST_OBJ = ['знак', 'мусор']
FONT = cv2.FONT_HERSHEY_COMPLEX
WEIGHTS = 'PATH_YOUR_MODEL'
MODEL_YOLO = torch.hub.load('ultralytics/yolov5', 'custom', WEIGHTS, device=DEVICE, force_reload=True)
MODEL_YOLO.conf = 0.8

with open('token.txt', 'r') as f:
    TOKEN = f.read()


def start(update, context):
    update.message.reply_text('Приветствую! Я бот для нахождения дорожных знаков и мусора для СКПДИ. '
                              'Отправь фото или видео и получи ответ')


def help(update, context):
    update.message.reply_text("""
    /start - Вступительная информация
    /help - Показать это сообщение
    """)


def handle_message(update, context):
    update.message.reply_text('Отправь фото')


def handle_photo(update, context):
    start_time = datetime.now()
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = list(MODEL_YOLO(img).pred[0].numpy())
    objs = []
    if len(result) == 0:
        update.message.reply_text('На фото не обнаруженно дорожных знаков или мусора')
    else:
        for obj in result:
            x_up, y_up, x_low, y_low = obj[:4]
            variance = round(obj[4] + 0, 2)
            id_ = int(obj[5])
            objs.append(LIST_OBJ[id_])
            text = f'{LIST_OBJ[id_]} - {variance}'
            text_size, _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
            text_w, text_h = text_size
            img_response = cv2.rectangle(img, (int(x_up), int(y_up - text_h)), (int(x_low), int(y_up)), (0, 0, 255), -1)
            img_response = cv2.rectangle(img_response, (int(x_up), int(y_up)),
                                         (int(x_low), int(y_low)), (0, 0, 255), 2)
            img_response = cv2.putText(img_response, text, (int(x_up), int(y_up - (img.shape[0] // 100))),
                                       FONT, FONT_SCALE, (255, 0, 0), THICKNESS)

        pil_image = Image.fromarray(img_response)
        bio = BytesIO()
        pil_image.save(bio, 'JPEG')
        bio.seek(0)

        context.bot.send_photo(update.message.chat.id, bio)
        # context.bot.send_video(chat_id=update.message.chat_id, video=open('/home/artemii_vibs/PycharmProjects/tg_bot_skpdi/file_16.mp4_new.mp4', 'rb'), supports_streaming=True)
        context.bot.send_message(update.message.chat.id, f"""
    На фото обнаруженны:
    Дорожные знаки - {objs.count(LIST_OBJ[0])}
    Мусор - {objs.count(LIST_OBJ[1])}
        """)
    time_answer = ((datetime.now() - start_time).microseconds) / 1000000
    logging.info(f"""
        
Completed send photo with detect: 
Дорожные знаки - {objs.count(LIST_OBJ[0])} 
Мусор - {objs.count(LIST_OBJ[1])}
Send answer to  {update.message.from_user.id}
                {update.message.from_user.first_name}
                {update.message.from_user.last_name}
                {update.message.from_user.username}
Time answer seconds - {time_answer}

""")


def handle_video(update, context):
    start_time = datetime.now()
    file = context.bot.get_file(update.message.video).download()
    cap = cv2.VideoCapture(file)
    count_frames_video = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    fps_video = round(cap.get(cv2.CAP_PROP_FPS))
    name_video = f'{file}_new.mp4'
    out = cv2.VideoWriter(name_video, cv2.VideoWriter_fourcc(*'DIVX'), fps_video, size, 0)
    for _ in tqdm(range(count_frames_video), desc=''):
        success, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        result = list(MODEL_YOLO(img).pred[0].numpy())
        for obj in result:
            x_up, y_up, x_low, y_low = obj[:4]
            variance = round(obj[4] + 0, 2)
            id_ = int(obj[5])
            text = f'{LIST_OBJ[id_]} - {variance}'
            text_size, _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
            text_w, text_h = text_size
            img = cv2.rectangle(img, (int(x_up), int(y_up - text_h)), (int(x_low), int(y_up)), (0, 0, 255), -1)
            img = cv2.rectangle(img, (int(x_up), int(y_up)),
                                         (int(x_low), int(y_low)), (0, 0, 255), 2)
            img = cv2.putText(img, text, (int(x_up), int(y_up - (img.shape[0] // 100))),
                                       FONT, FONT_SCALE, (255, 0, 0), THICKNESS)

        out.write(img)
    out.release()
    context.bot.send_video(chat_id=update.message.chat_id, video=open(name_video, 'rb'), supports_streaming=True)



updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler('start', start))
dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))
dp.add_handler(MessageHandler(Filters.video, handle_video))

updater.start_polling()
updater.idle()
