# 🚀 Инструкция по развертыванию

## Локальный запуск

### 1. Подготовка окружения

```bash
# Клонируйте репозиторий
git clone <your-repo-url>
cd Signature_Verification

# Создайте виртуальное окружение
python -m venv venv

# Активируйте окружение
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Установите зависимости
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
# Запустите Jupyter notebook
jupyter notebook Signature_Verification.ipynb

# Выполните все ячейки для обучения модели
# Убедитесь, что создался файл signature_verification_model_fixed.h5
```

### 3. Запуск приложения

```bash
# Простая версия
streamlit run app.py

# Расширенная версия с аналитикой
streamlit run app_advanced.py
```

## Развертывание на Streamlit Cloud

### 1. Подготовка репозитория

1. **Создайте GitHub репозиторий:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/signature-verification.git
   git push -u origin main
   ```

2. **Убедитесь, что включены файлы:**
   - `app.py` или `app_advanced.py`
   - `requirements.txt`
   - `signature_verification_model_fixed.h5` (обученная модель)
   - `.streamlit/config.toml`

### 2. Создание приложения на Streamlit Cloud

1. Перейдите на [share.streamlit.io](https://share.streamlit.io)
2. Нажмите "New app"
3. Заполните форму:
   - **Repository:** yourusername/signature-verification
   - **Branch:** main
   - **Main file path:** app.py (или app_advanced.py)
4. Нажмите "Deploy!"

### 3. Настройка переменных окружения (если нужно)

В настройках приложения добавьте:
- `TF_CPP_MIN_LOG_LEVEL=2` (для уменьшения логов TensorFlow)

## Развертывание на GitHub Pages (статическая версия)

### 1. Установка streamlit-static

```bash
pip install streamlit-static
```

### 2. Создание статической версии

```bash
# Создайте статическую версию
streamlit build app.py

# Или для расширенной версии
streamlit build app_advanced.py
```

### 3. Настройка GitHub Pages

1. В настройках репозитория включите GitHub Pages
2. Выберите источник: "GitHub Actions"
3. Создайте файл `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install streamlit-static
    
    - name: Build static site
      run: streamlit build app.py
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./build
```

## Развертывание на Heroku

### 1. Создайте Procfile

```bash
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
```

### 2. Создайте runtime.txt

```bash
echo "python-3.9.18" > runtime.txt
```

### 3. Развертывание

```bash
# Установите Heroku CLI
# Создайте приложение
heroku create your-app-name

# Разверните
git push heroku main
```

## Развертывание на Docker

### 1. Создайте Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Сборка и запуск

```bash
# Сборка образа
docker build -t signature-verification .

# Запуск контейнера
docker run -p 8501:8501 signature-verification
```

## Мониторинг и логи

### Streamlit Cloud
- Логи доступны в панели управления Streamlit Cloud
- Автоматическое обновление при изменениях в репозитории

### Heroku
```bash
# Просмотр логов
heroku logs --tail

# Масштабирование
heroku ps:scale web=1
```

### Docker
```bash
# Просмотр логов
docker logs <container-id>

# Мониторинг ресурсов
docker stats
```

## Безопасность

### Рекомендации:
1. **Не загружайте модель в публичный репозиторий** - используйте Git LFS или приватные репозитории
2. **Ограничьте доступ** - настройте аутентификацию для продакшена
3. **Мониторинг** - отслеживайте использование и производительность
4. **Резервное копирование** - регулярно создавайте бэкапы модели

### Переменные окружения:
```bash
# Для продакшена
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

## Устранение неполадок

### Частые проблемы:

1. **Модель не загружается:**
   - Проверьте путь к файлу модели
   - Убедитесь, что файл не поврежден
   - Проверьте версию TensorFlow

2. **Ошибки памяти:**
   - Уменьшите размер изображений
   - Используйте более легкую модель
   - Настройте лимиты памяти

3. **Медленная работа:**
   - Используйте GPU если доступен
   - Оптимизируйте предобработку
   - Кэшируйте результаты

### Логи для диагностики:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

📞 **Поддержка:** DeepSign Support







