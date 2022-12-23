from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import cv2
from PIL import Image
import os
import shutil
import numpy as np
import random
import string
from face import Face


model = Face()
first_images = {}
busy = {}

def generate_random_string(length):
  # Choose a random selection of characters from the ascii letters and digits
  characters = string.ascii_letters + string.digits
  return ''.join(random.choices(characters, k=length))

def process_image(update, context):
  # Get the image from the update
  image = update.message.photo[-1]

  id = update.message.from_user.id
  chat_id = update.message.chat_id
  if chat_id in busy and busy[chat_id]:
    context.bot.send_message(
        chat_id=chat_id,
        text='Your previous request is still being processed. please wait a moment..'
    )
  else:
    if not os.path.isdir(f'bot/{id}'):
        os.mkdir(f'bot/{id}')
    if os.path.isdir(f'bot/{id}/{chat_id}') and len(os.listdir(f'bot/{id}/{chat_id}'))>0 and chat_id not in first_images:
        new_path = f'bot/{id}/{chat_id}_{generate_random_string(6)}'
        while os.path.isdir(new_path):
            new_path = f'bot/{id}/{chat_id}_{generate_random_string(6)}'
        shutil.move(f'bot/{id}/{chat_id}', new_path)
    if not os.path.isdir(f'bot/{id}/{chat_id}'):
        os.mkdir(f'bot/{id}/{chat_id}')
    f = image.get_file()
    if chat_id not in first_images:
        context.bot.send_message(
            chat_id=chat_id,
            text='Processing please wait a moment..'
        )
        p = f.download(f'bot/{id}/{chat_id}/file_1.png')
        busy[chat_id] = True
        res = model.embedding(p)
        if not res[0]:
            os.unlink(p)
            context.bot.send_message(
                chat_id=chat_id,
                text='No human face found. please resend first image'
            )
            busy[chat_id] = False
        else:
            first_images[chat_id] = res[2]
            res[1].save(f'bot/{id}/{chat_id}/face_1.png')
            with open(f'bot/{id}/{chat_id}/face_1.png','rb') as f1:
                context.bot.send_photo(
                    chat_id=update.message.chat_id,
                    photo=f1,
                    caption='Face found on first image'
                )
                context.bot.send_message(
                    chat_id=chat_id,
                    text='Please send me the second image'
                )
            busy[chat_id] = False
    else:
        context.bot.send_message(
            chat_id=chat_id,
            text='Processing please wait a moment..'
        )
        p = f.download(f'bot/{id}/{chat_id}/file_2.png')
        busy[chat_id] = True
        res = model.embedding(p)
        if not res[0]:
            os.unlink(p)
            context.bot.send_message(
                chat_id=chat_id,
                text='No human face found. please resend second image'
            )
            busy[chat_id] = False
        else:
            res[1].save(f'bot/{id}/{chat_id}/face_2.png')
            with open(f'bot/{id}/{chat_id}/face_2.png','rb') as f2:
                context.bot.send_photo(
                    chat_id=update.message.chat_id,
                    photo=f2,
                    caption='Face found on second image'
                )
                distance = np.linalg.norm(first_images[chat_id]-res[2])
                emoji = {
                    "happy":  ["\U0001F600", "\U0001F60D", "\U0001F60E", "\U0001F929"],
                    "sad": ["\U0001F923", "\U0001F625", "\U0001F62A", "\U0001F61D"]}
                percentage = (100-random.randint(0,5)) if distance<1e-2 else (10-random.randint(0,5)) if distance>1 else -100*distance+110
                state = 'happy' if percentage>75 else 'sad'
                context.bot.send_message(
                    chat_id=chat_id,
                    text=f"Person 1 and Person 2's Match probability is {round(percentage)}% {emoji[state][random.randint(0,3)]}"
                )
                busy[chat_id] = False
                del first_images[chat_id]

def image_command(update, context):
  # Get the user's message
  message = update.message.text

  # Check if the message is the '/start' command
  if message == '/start':
    # Send a message to the user asking them to send an start
    context.bot.send_message(
        chat_id=update.message.chat_id,
        text='Please send me the first image'
    )

def reset_session(update, context):
  # Get the user's message
  message = update.message.text
  chat_id = update.message.chat_id

  # Check if the message is the '/reset' command
  if message == '/reset':
    if chat_id in busy:
        busy[chat_id] = False
    if chat_id in first_images:
        del first_images[chat_id]
    # Inform user session was reset
    context.bot.send_message(
        chat_id=update.message.chat_id,
        text='Session reset'
    )

def display_help(update, context):
  message = update.message.text
  # Check if the message is the '/help' command
  if message == '/help':
    # Inform user session was reset
    context.bot.send_message(
        chat_id=update.message.chat_id,
        text="This bot calculates the match percentage between two people from images! [N.B: It's only for fun, nothing real\U0001F925]"
    )

# Set up the updater and dispatcher
updater = Updater(token='5780289102:AAHp_VO-lCAmWtwc-uYWgCk-sCPanQB0Ba0', use_context=True)
dispatcher = updater.dispatcher

# Add a message handler for image messages
image_handler = MessageHandler(Filters.photo, process_image)
dispatcher.add_handler(image_handler)

# Add a command handler for the '/image' command
command_handler = CommandHandler('start', image_command)
dispatcher.add_handler(command_handler)

# Add reset handler for the '/reset' command
reset_session = CommandHandler('reset', reset_session)
dispatcher.add_handler(reset_session)

# Add reset handler for the '/help' command
display_help = CommandHandler('help', display_help)
dispatcher.add_handler(display_help)

# Start the bot
updater.start_polling()
while True:
    pass
