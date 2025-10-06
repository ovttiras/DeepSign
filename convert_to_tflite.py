#!/usr/bin/env python3
"""
Скрипт для конвертации H5 модели в TensorFlow Lite формат
Использование: python convert_to_tflite.py
"""

import tensorflow as tf
import os

def convert_h5_to_tflite(h5_path, tflite_path):
    """Конвертирует H5 модель в TensorFlow Lite формат"""
    try:
        print(f"🔄 Загружаем модель из {h5_path}...")
        
        # Загружаем H5 модель
        model = tf.keras.models.load_model(h5_path)
        
        print("✅ Модель загружена успешно")
        print(f"📊 Архитектура модели:")
        model.summary()
        
        # Создаем конвертер
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Настройки оптимизации
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Конвертируем
        print("🔄 Конвертируем в TensorFlow Lite...")
        tflite_model = converter.convert()
        
        # Сохраняем
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Получаем размеры файлов
        h5_size = os.path.getsize(h5_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
        
        print(f"✅ Конвертация завершена!")
        print(f"📁 H5 модель: {h5_size:.2f} MB")
        print(f"📁 TFLite модель: {tflite_size:.2f} MB")
        print(f"📉 Сжатие: {((h5_size - tflite_size) / h5_size * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
        return False

def main():
    """Основная функция"""
    print("🚀 Конвертер H5 → TensorFlow Lite")
    print("=" * 50)
    
    # Ищем H5 модели
    h5_files = [
        'signature_verification_model_fixed.h5',
        'signature_verification_model.h5'
    ]
    
    found_models = []
    for file in h5_files:
        if os.path.exists(file):
            found_models.append(file)
    
    if not found_models:
        print("❌ H5 модели не найдены!")
        print("Доступные файлы:")
        for file in os.listdir('.'):
            if file.endswith('.h5'):
                print(f"  - {file}")
        return
    
    print(f"📋 Найдено H5 моделей: {len(found_models)}")
    for i, model in enumerate(found_models, 1):
        print(f"  {i}. {model}")
    
    # Конвертируем все найденные модели
    for h5_file in found_models:
        print(f"\n🔄 Конвертируем {h5_file}...")
        
        # Создаем имя для TFLite файла
        tflite_file = h5_file.replace('.h5', '.tflite')
        
        # Если уже есть TFLite версия, спрашиваем
        if os.path.exists(tflite_file):
            response = input(f"⚠️  {tflite_file} уже существует. Перезаписать? (y/n): ")
            if response.lower() != 'y':
                print("⏭️  Пропускаем...")
                continue
        
        # Конвертируем
        success = convert_h5_to_tflite(h5_file, tflite_file)
        
        if success:
            print(f"✅ {h5_file} → {tflite_file}")
        else:
            print(f"❌ Ошибка конвертации {h5_file}")
    
    print("\n🎉 Конвертация завершена!")
    print("\n📋 Рекомендации:")
    print("1. Протестируйте TFLite модель в приложении")
    print("2. Убедитесь, что точность не изменилась")
    print("3. При необходимости удалите H5 файлы для экономии места")

if __name__ == "__main__":
    main()
