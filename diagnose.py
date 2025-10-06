#!/usr/bin/env python3
"""
Скрипт диагностики для DeepSign
Проверяет все необходимые компоненты для запуска приложения
"""

import sys
import os
import subprocess

def check_python():
    """Проверка версии Python"""
    print("🐍 Проверка Python...")
    version = sys.version_info
    print(f"   Версия Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Требуется Python 3.8 или выше!")
        return False
    else:
        print("   ✅ Python версия подходит")
        return True

def check_package(package_name, import_name=None):
    """Проверка установки пакета"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'неизвестно')
        print(f"   ✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"   ❌ {package_name}: не установлен")
        return False

def check_dependencies():
    """Проверка всех зависимостей"""
    print("\n📦 Проверка зависимостей...")
    
    packages = [
        ('streamlit', 'streamlit'),
        ('tensorflow', 'tensorflow'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('Pillow', 'PIL'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('plotly', 'plotly'),
        ('pandas', 'pandas')
    ]
    
    all_ok = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            all_ok = False
    
    return all_ok

def check_model_files():
    """Проверка наличия файлов модели"""
    print("\n🤖 Проверка модели...")
    
    model_files = [
        'signature_verification_model_fixed.h5',
        'signature_verification_model.h5'
    ]
    
    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"   ✅ {model_file}: {size:.1f} MB")
            found_models.append(model_file)
        else:
            print(f"   ❌ {model_file}: не найден")
    
    if not found_models:
        print("   ⚠️  Модели не найдены! Будет создана тестовая модель")
        return False
    else:
        return True

def check_app_files():
    """Проверка основных файлов приложения"""
    print("\n📁 Проверка файлов приложения...")
    
    app_files = [
        'app.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_ok = True
    for file in app_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}: не найден")
            all_ok = False
    
    return all_ok

def test_streamlit():
    """Тест запуска Streamlit"""
    print("\n🚀 Тест запуска Streamlit...")
    
    try:
        # Попробуем импортировать streamlit
        import streamlit as st
        print("   ✅ Streamlit импортируется успешно")
        
        # Проверим версию
        print(f"   📋 Версия Streamlit: {st.__version__}")
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка импорта Streamlit: {e}")
        return False

def main():
    """Основная функция диагностики"""
    print("🔍 Диагностика DeepSign")
    print("=" * 50)
    
    # Проверки
    python_ok = check_python()
    deps_ok = check_dependencies()
    model_ok = check_model_files()
    files_ok = check_app_files()
    streamlit_ok = test_streamlit()
    
    # Итоговый отчет
    print("\n📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 50)
    
    if python_ok and deps_ok and files_ok and streamlit_ok:
        print("✅ Все проверки пройдены! Приложение должно запуститься.")
        print("\n🚀 Для запуска выполните:")
        print("   streamlit run app.py")
        print("   или")
        print("   run_app.bat")
    else:
        print("❌ Обнаружены проблемы:")
        
        if not python_ok:
            print("   - Установите Python 3.8+")
        
        if not deps_ok:
            print("   - Установите зависимости: pip install -r requirements.txt")
        
        if not files_ok:
            print("   - Проверьте наличие файлов приложения")
        
        if not streamlit_ok:
            print("   - Переустановите Streamlit: pip install streamlit")
        
        if not model_ok:
            print("   - Запустите Signature_Verification.ipynb для обучения модели")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()

