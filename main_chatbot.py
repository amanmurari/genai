import chainlit as cl
from openai import OpenAI
import os
import base64
import json

from pydantic_core.core_schema import none_schema

os.environ[
    "OPENAI_API_KEY"] = "aman"
client = OpenAI()


def image_text(image_url):
  bse = None
  with open(image_url, "rb") as f:
    bse = base64.b64encode(f.read())
    bse = bse.decode('utf-8')
  return bse


def audio_text(audio_url):
  audio_file = open(audio_url, "rb")
  transcription = client.audio.transcriptions.create(model="whisper-1",
                                                     file=audio_file)

  return transcription.text


def append_messages(text=None, image_url=None, audio=None):
  message_list = []
  if image_url:
    message_list.append({"type": "image_url", "image_url": {"url": image_url}})
  if text and not audio:
    message_list.append({"type": "text", "text": text})
  if audio:
    message_list.append({"type": "text", "text": text + "\n" + audio})

  response = client.chat.completions.create(model="gpt-4o",
                                            messages=[{
                                                "role": "user",
                                                "content": message_list
                                            }],
                                            max_tokens=1024)
  return response.choices[0]


@cl.on_message
async def main(msg: cl.Message):
  a_text=None
  response=None
  images = [file for file in msg.elements if "image" in file.mime]

  audios = [file for file in msg.elements if "audio" in file.mime]
  if len(images) > 0:
    base64_image = image_text(images[0].path)
    image_url = f"data:image/png;base64,{base64_image}"

  elif len(audios) > 0:
    a_text = audio_text(audios[0].path)

  
  response_msg = cl.Message(content="")

  if len(images) == 0 and len(audios) == 0:
    response = append_messages(text=msg.content)

  elif len(audios) == 0:
    response = append_messages(image_url=image_url,text=msg.content)
  if len(images) == 0 and len(audios) == 0:

    response = append_messages(text=msg.content)

  elif len(audios) == 0:
    response = append_messages(image_url=image_url, text=msg.content)

  else:
    response = append_messages(text=msg.content, audio=a_text)

  response_msg.content = response.message.content
  await response_msg.send()