import os
os.environ["http_proxy"] = "http://localhost:7897"
os.environ["https_proxy"] = "http://localhost:7897"

from audio_processing import transcribe_audio
from image_processing import encode_image
from api_requests import send_image_query
from image_detection import detect_objects

# 示例路径
audio_file_path = "speech.mp3"
image_path = "figure.webp"

# 处理音频
transcription = transcribe_audio(audio_file_path)
print("语音转文字:", transcription)

# 编码图片
encoded_image = encode_image(image_path)

# 发送图像查询
response = send_image_query(encoded_image)
print("图片理解:", response['choices'][0]['message']['content'])

# 对象检测
image_path = "figure.webp"
query0 = "苹果"
query1 = "盘子"
results = detect_objects(image_path, query0, query1)

