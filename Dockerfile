# smaller images like alpine do not work for tensorflow, unfortunately.
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD [ "streamlit", "run","app.py" ]