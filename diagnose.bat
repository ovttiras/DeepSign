@echo off
echo Диагностика DeepSign
echo ====================

echo.
echo 1. Проверка Python...
python --version
if errorlevel 1 (
    echo ОШИБКА: Python не найден!
    goto :end
)

echo.
echo 2. Проверка pip...
pip --version
if errorlevel 1 (
    echo ОШИБКА: pip не найден!
    goto :end
)

echo.
echo 3. Установка зависимостей...
pip install -r requirements.txt

echo.
echo 4. Проверка Streamlit...
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
if errorlevel 1 (
    echo ОШИБКА: Streamlit не установлен!
    goto :end
)

echo.
echo 5. Проверка TensorFlow...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
if errorlevel 1 (
    echo ОШИБКА: TensorFlow не установлен!
    goto :end
)

echo.
echo 6. Проверка файлов модели...
if exist "model.tflite" (
    echo Найдена оптимизированная модель: model.tflite (TensorFlow Lite)
) else if exist "signature_verification_model_fixed.h5" (
    echo Найдена модель: signature_verification_model_fixed.h5
) else if exist "signature_verification_model.h5" (
    echo Найдена модель: signature_verification_model.h5
) else (
    echo ПРЕДУПРЕЖДЕНИЕ: Модель не найдена!
)

echo.
echo 7. Тест запуска приложения...
echo Запуск тестового приложения...
streamlit run app_test.py

:end
echo.
echo Диагностика завершена.
pause

