FROM python:3.9
CMD mkdir /app
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD streamlit run app.py
