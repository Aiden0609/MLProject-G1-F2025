FROM python:3.12.1

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y cdo
RUN pip install --no-cache-dir -r requirements.txt

copy .cdsapirc /root/

COPY . .

CMD [ "python", "./fetch_data.py" ]