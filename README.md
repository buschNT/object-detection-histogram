# Object Detection (histogram)

Fastapi with object detection endpoint.

Select an image [jpg] using the file selection on the homepage and press the upload button.

## Run with Docker

This app can be run using docker using the following commands below. Just be aware that because of the torch dependencies, the docker build takes quite some time and results in a large image (~10GB).

```bash
docker build . -t object-detection
docker run -p 8000:8000 object-detection
```

Access the app in your local browser [http://0.0.0.0:8000](http://0.0.0.0:8000) or [http://localhost:8000](http://localhost:8000).

## Develop using Virtual Environment

Create a virtual environment and active it afterwards (activation depends on):

```bash
python3.12 -m venv .venv
```

[Optional] Update pip:

```bash
pip install --upgrade pip
```

Install the required dependencies using [poetry](https://python-poetry.org/):

```bash
poetry install
```

Change into the app folder and start the server:

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```
