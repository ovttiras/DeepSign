@echo off
echo Запуск DeepSign - Система верификации подписей...
echo.

REM Проверка наличия Python
echo Проверка Python...
python --version
if errorlevel 1 (
    echo ОШИБКА: Python не найден! Установите Python 3.8+ и добавьте в PATH
    pause
    exit /b 1
)

REM Установка зависимостей
echo.
echo Установка зависимостей...
pip install -r requirements.txt
if errorlevel 1 (
    echo ОШИБКА: Не удалось установить зависимости!
    pause
    exit /b 1
)

REM Проверка наличия модели
echo.
echo Проверка модели...
if exist "model.tflite" (
    echo Найдена оптимизированная модель: model.tflite (TensorFlow Lite)
) else if exist "signature_verification_model_fixed.h5" (
    echo Найдена исправленная модель: signature_verification_model_fixed.h5
) else if exist "signature_verification_model.h5" (
    echo Найдена базовая модель: signature_verification_model.h5
) else (
    echo ПРЕДУПРЕЖДЕНИЕ: Модель не найдена!
    echo Будет создана тестовая модель для демонстрации
    echo Для полной функциональности запустите Signature_Verification.ipynb
    echo.
)

REM Запуск приложения
echo.
echo Запуск веб-приложения DeepSign...
echo Откройте браузер по адресу: http://localhost:8501
echo.
echo Для остановки нажмите Ctrl+C
echo.

streamlit run app.py

echo.
echo Приложение остановлено.
pause