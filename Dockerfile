FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements_docker.txt

EXPOSE 8501

ENV MODEL_PATH=/training/models/trained.h5

CMD ["streamlit" , "run" , "app.py"]




