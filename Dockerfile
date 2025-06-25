# Etapa 1: imagem base mínima com Python
FROM python:3.10-slim as base

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema que torch e outras libs precisam
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Copia apenas arquivos necessários
COPY requirements.txt .

# Instala dependências com cache
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do app
COPY . .

# Expõe a porta usada pelo Flask
EXPOSE 5000

# Comando de inicialização
CMD ["python", "app.py"]
