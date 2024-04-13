# 
FROM python:3.9

# 
COPY requirements.txt /app/requirements.txt

WORKDIR /app

# 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
# 


# 
COPY . /app/

# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]