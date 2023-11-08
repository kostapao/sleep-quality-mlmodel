FROM python:3.8.12-slim

RUN pip install pipenv
RUN pip install numpy
RUN pip install scipy

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["main.py", "modelrf_n_est_200_maxdepth_5_minleaf_8.bin", "./"]

EXPOSE 8080

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]