FROM python:3.8-slim-buster

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD streamlit run --server.port 8080 --server.enableCORS false app.py