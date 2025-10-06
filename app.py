import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
from PIL import Image
import base64

# Настройка страницы
st.set_page_config(
    page_title="DeepSign - Система верификации подписей",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/deepsign',
        'Report a bug': "https://github.com/yourusername/deepsign/issues",
        'About': "DeepSign - Инновационная система верификации подписей с использованием ИИ"
    }
)

# Современные CSS стили с лучшими практиками UX/UI
st.markdown("""
<style>
    /* CSS переменные в стиле Агропромбанка */
    :root {
        --primary-color: #1E3A8A;
        --primary-dark: #1E40AF;
        --primary-light: #3B82F6;
        --accent-color: #F59E0B;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --text-primary: #1F2937;
        --text-secondary: #6B7280;
        --background: #F8FAFC;
        --surface: #FFFFFF;
        --border: #E5E7EB;
        --shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
        --shadow-hover: 0 4px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06);
        --border-radius: 8px;
        --border-radius-small: 6px;
        --spacing-xs: 0.25rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
        --spacing-xl: 2rem;
        --spacing-xxl: 3rem;
    }
    
    /* Базовые стили */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Заголовок в стиле Агропромбанка */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        padding: var(--spacing-xxl);
        border-radius: var(--border-radius);
        margin-bottom: var(--spacing-xl);
        text-align: center;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.05) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        margin: 0;
        font-weight: 600;
        position: relative;
        z-index: 1;
        letter-spacing: -0.01em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1rem;
        margin: var(--spacing-sm) 0 0 0;
        opacity: 0.9;
        position: relative;
        z-index: 1;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Карточки в стиле Агропромбанка */
    .card {
        background: var(--surface);
        padding: var(--spacing-lg);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        margin-bottom: var(--spacing-md);
        border: 1px solid var(--border);
        transition: all 0.2s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .card:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-1px);
        border-color: var(--primary-light);
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, var(--primary-color) 0%, var(--primary-light) 100%);
    }
    
    .card h3 {
        color: var(--primary-color);
        margin: 0 0 var(--spacing-sm) 0;
        font-size: 1.2rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
    }
    
    .card p {
        margin: var(--spacing-sm) 0;
        line-height: 1.6;
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    .card ul {
        margin: var(--spacing-sm) 0;
        padding-left: var(--spacing-lg);
    }
    
    .card li {
        margin: var(--spacing-xs) 0;
        line-height: 1.6;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    
    /* Кнопки в стиле Агропромбанка */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: var(--border-radius-small);
        padding: var(--spacing-sm) var(--spacing-lg);
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease-in-out;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
        text-transform: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-hover);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow);
    }
    
    /* Файловый загрузчик в стиле Агропромбанка */
    .stFileUploader > div {
        border: 2px dashed var(--primary-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-xl);
        text-align: center;
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        transition: all 0.2s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-dark);
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        transform: translateY(-1px);
        box-shadow: var(--shadow);
    }
    
    .stFileUploader > div::before {
        content: '📁';
        font-size: 1.8rem;
        display: block;
        margin-bottom: var(--spacing-sm);
        opacity: 0.8;
    }
    
    /* Результаты в стиле Агропромбанка */
    .result-success {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid var(--success-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-lg);
        margin: var(--spacing-md) 0;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    .result-success::before {
        content: '✅';
        position: absolute;
        top: var(--spacing-md);
        right: var(--spacing-md);
        font-size: 1.4rem;
    }
    
    .result-warning {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 1px solid var(--warning-color);
        border-radius: var(--border-radius);
        padding: var(--spacing-lg);
        margin: var(--spacing-md) 0;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    .result-warning::before {
        content: '⚠️';
        position: absolute;
        top: var(--spacing-md);
        right: var(--spacing-md);
        font-size: 1.4rem;
    }
    
    /* Улучшенный прогресс бар */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-light) 100%);
        border-radius: var(--border-radius-small);
    }
    
    /* Улучшенный сайдбар */
    .css-1d391kg {
        background: linear-gradient(180deg, #F8F9FA 0%, #E9ECEF 100%);
        border-right: 1px solid var(--border);
    }
    
    /* Адаптивность */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .card {
            padding: var(--spacing-md);
        }
        
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* Анимации загрузки */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Улучшенные уведомления */
    .stAlert {
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--shadow);
    }
</style>
""", unsafe_allow_html=True)

# Функция предобработки изображения
def preprocess_image(image):
    """Предобработка изображения для модели"""
    try:
        # Конвертируем PIL в numpy array
        img_array = np.array(image)
        
        # Конвертируем в grayscale если нужно
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Изменяем размер
        img_array = cv2.resize(img_array, (220, 155))
        
        # Бинаризация
        _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Нормализация и конвертация в FLOAT32
        img_array = (img_array / 255.0).astype(np.float32)
        
        # Добавляем канал
        img_array = img_array.reshape(1, 155, 220, 1)
        
        return img_array
    except Exception as e:
        st.error(f"Ошибка предобработки изображения: {str(e)}")
        return None

# Функция предсказания
def predict_signature(model, image, model_format="keras", threshold=0.5):
    """Предсказание подлинности подписи"""
    try:
        if model_format == "tflite":
            # Предсказание для TensorFlow Lite модели
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            # Убеждаемся, что данные имеют правильный тип (FLOAT32)
            image_float32 = image.astype(np.float32)
            
            # Устанавливаем входные данные
            model.set_tensor(input_details[0]['index'], image_float32)
            
            # Запускаем инференс
            model.invoke()
            
            # Получаем результат
            prediction = model.get_tensor(output_details[0]['index'])[0][0]
            
        else:
            # Предсказание для Keras модели
            prediction = model.predict(image, verbose=0)[0][0]
        
        # Определяем тип результата с учетом порога
        if prediction > threshold:
            result = "Подлинная подпись"
            result_type = "genuine"
        else:
            result = "Поддельная подпись"
            result_type = "forged"
        
        confidence = max(prediction, 1 - prediction)
        
        return result, confidence, result_type
        
    except Exception as e:
        st.error(f"Ошибка предсказания: {str(e)}")
        return None, None, None

# Функция загрузки модели TensorFlow Lite
@st.cache_resource
def load_signature_model():
    """Загружает предобученную модель TensorFlow Lite для верификации подписей"""
    try:
        # Проверяем наличие файлов модели (приоритет TensorFlow Lite)
        model_files = [
            'model.tflite',
            'signature_verification_model_fixed.h5',
            'signature_verification_model.h5'
        ]
        
        model_path = None
        model_type = None
        
        for file in model_files:
            if os.path.exists(file):
                model_path = file
                if file.endswith('.tflite'):
                    model_type = "TensorFlow Lite модель"
                elif "fixed" in file:
                    model_type = "Исправленная H5 модель"
                else:
                    model_type = "Базовая H5 модель"
                break
        
        if model_path is None:
            st.warning("⚠️ Модель не найдена! Создайте тестовую модель...")
            # Создаем простую тестовую модель
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
            
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(155, 220, 1)),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model, "Тестовая модель (не обучена)", "keras"
        
        # Загружаем модель в зависимости от типа
        if model_path.endswith('.tflite'):
            # Загружаем TensorFlow Lite модель
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter, model_type, "tflite"
        else:
            # Загружаем H5 модель (для совместимости)
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            return model, model_type, "keras"
        
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        st.info("Попробуйте запустить Signature_Verification.ipynb для обучения модели")
        return None, None, None

# Загрузка модели
model, model_type, model_format = load_signature_model()

if model is not None:
    # Заголовок приложения
    st.markdown("""
    <div class="main-header">
        <h1>DeepSign</h1>
        <p>Инновационная технология распознавания подлинности подписей на примере базы данных CEDAR</p>
    </div>
    """, unsafe_allow_html=True)

    # Информационная секция о системе
    st.markdown("### ℹ️ О системе верификации подписей")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>🧠 Технология</h3>
            <p>Используем глубокое обучение и компьютерное зрение для анализа уникальных характеристик подписей.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>⚡ Скорость</h3>
            <p>Анализ подписи занимает менее 1 секунды, обеспечивая быструю обработку документов.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h3>🎯 Точность</h3>
            <p>Точность на тестовой выборке составляет 79%, что обеспечивает надежную защиту от подделок.</p>
        </div>
        """, unsafe_allow_html=True)

    # Информация о базе данных CEDAR
    st.markdown("---")
    st.markdown("### 📊 База данных CEDAR")
    st.markdown("""
    <div class="card">
        <p><strong>Источник:</strong> CEDAR (Center of Excellence for Document Analysis and Recognition)</p>
        <p><strong>Ссылка на датасет:</strong> <a href="https://cedar.buffalo.edu/NIJ/data/signatures.rar" target="_blank">https://cedar.buffalo.edu/NIJ/data/signatures.rar</a></p>
        <p><strong>Размер:</strong> 55 авторов, 24 подписи каждого</p>
        <p><strong>Общее количество подписей:</strong> 2640</p>
        <p><strong>Распределение:</strong> Равномерно представлены двумя классами - настоящие и поддельные подписи</p>
        <p><strong>Формат:</strong> Отсканированные изображения высокого качества</p>
        <p><strong>Особенности:</strong> Различные стили подписей, естественные вариации</p>
    </div>
    """, unsafe_allow_html=True)

    # Инструкция по использованию приложения
    st.markdown("---")
    st.markdown("### 📋 Инструкция по использованию приложения")
    st.markdown("""
    <div class="card">
        <ol>
            <li><strong>Скачайте RAR архив с официальной страницы CEDAR:</strong> по ссылке <a href="https://cedar.buffalo.edu/NIJ/data/signatures.rar" target="_blank">https://cedar.buffalo.edu/NIJ/data/signatures.rar</a></li>
            <li><strong>Распакуйте архив:</strong> В архиве находятся две папки:
                <ul>
                    <li><strong>full_org</strong> - содержит подлинные подписи</li>
                    <li><strong>full_forg</strong> - содержит поддельные подписи</li>
                </ul>
            </li>
            <li><strong>Выберите изображение:</strong> Выберите любое изображение из любой папки</li>
            <li><strong>Переименуйте файл:</strong> Для объективности проверки произвольно переименуйте выбранный файл</li>
            <li><strong>Загрузите в приложение:</strong> Используйте блок "📤 Загрузка подписи" ниже для загрузки выбранного изображения подписи</li>
            <li><strong>Анализ подписи:</strong> После загрузки изображения нажмите кнопку "Анализировать подпись"</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Блок загрузки подписи
    st.markdown("---")
    st.markdown("### 🚀 Начните анализ подписи")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📤 Загрузка подписи")
        st.markdown("Загрузите изображение подписи для анализа. Поддерживаются форматы: PNG, JPG, JPEG, BMP")
        
        uploaded_file = st.file_uploader(
            "Выберите изображение подписи",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Поддерживаемые форматы: PNG, JPG, JPEG, BMP"
        )
        
        if uploaded_file is not None:
            # Отображение загруженного изображения
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженная подпись", use_container_width=True)
            
            # Настройка порога классификации
            st.markdown("#### ⚙️ Настройки анализа")
            confidence_threshold = st.slider(
                "Порог классификации (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
                help="50% - стандартный порог\nВыше 50% - более строгая проверка (меньше ложных срабатываний)\nНиже 50% - более мягкая проверка (больше ложных срабатываний)"
            )
            
            # Кнопка анализа
            if st.button("🔍 Анализировать подпись", type="primary"):
                # Показываем индикатор загрузки
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 Загружаем изображение...")
                    progress_bar.progress(25)
                    
                    # Предобработка
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        status_text.text("🔍 Анализируем подпись...")
                        progress_bar.progress(50)
                        
                        # Предсказание
                        threshold = confidence_threshold / 100.0  # Конвертируем проценты в десятичную дробь
                        result, confidence, result_type = predict_signature(model, processed_image, model_format, threshold)
                        
                        status_text.text("✅ Анализ завершен!")
                        progress_bar.progress(100)
                        
                        # Небольшая задержка для UX
                        import time
                        time.sleep(0.5)
                        
                        # Очищаем индикаторы
                        progress_bar.empty()
                        status_text.empty()
                        
                        if result is not None:
                            # Отображение результата
                            st.markdown("#### 🎯 Результат анализа")
                            
                            # Анимированный результат
                            if result_type == "genuine":
                                st.markdown(f"""
                                <div class="result-success">
                                    <h2 style="margin: 0; color: #2E7D32;">✅ {result}</h2>
                                    <p style="margin: 0.5rem 0; font-size: 1.1rem; font-weight: 600;">Уверенность: {confidence:.1%}</p>
                                    <p style="margin: 0; color: #4CAF50; font-size: 0.9rem;">Подпись признана подлинной</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-warning">
                                    <h2 style="margin: 0; color: #F57C00;">❌ {result}</h2>
                                    <p style="margin: 0.5rem 0; font-size: 1.1rem; font-weight: 600;">Уверенность: {confidence:.1%}</p>
                                    <p style="margin: 0; color: #FF9800; font-size: 0.9rem;">Подпись признана поддельной</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        else:
                            st.error("❌ Ошибка при анализе подписи. Попробуйте другое изображение.")
                    else:
                        st.error("❌ Ошибка предобработки изображения. Проверьте формат файла.")
                        
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Произошла ошибка: {str(e)}")
                    st.info("💡 Попробуйте загрузить изображение в формате PNG, JPG, JPEG или BMP")

    # Детальная информация о модели и архитектуре
    st.markdown("---")
    st.markdown("### 🧠 Архитектура нейронной сети")

    st.markdown("""
    <div class="card">
        <h3>🔧 Технические детали</h3>
        <p><strong>Входные данные:</strong> Изображения 220×155 пикселей в градациях серого</p>
        <p><strong>Предобработка:</strong> Изменение размера, бинаризация, нормализация [0,1]</p>
        <p><strong>Архитектура:</strong> CNN с Conv2D, MaxPooling, Dropout, Dense слоями</p>
        <p><strong>Активация:</strong> Sigmoid для бинарной классификации</p>
        <p><strong>Обучение:</strong> Adam оптимизатор, Binary Crossentropy, валидация 20%</p>
        <p><strong>Производительность:</strong> Точность 79% (тест), время инференса < 1 сек</p>
        <p><strong>Совместимость:</strong> TensorFlow/Keras, поддержка TensorFlow Lite</p>
    </div>
    """, unsafe_allow_html=True)

    # Футер
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>© 2025 DeepSign was developed by Oleg Tinkov - Система верификации подписей</p>
        <p>Инновационные решения для банковской безопасности</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Не удалось загрузить модель. Проверьте наличие файла модели в папке приложения.")
    
    st.markdown("### 📁 Требуемые файлы:")
    st.code("""
    model.tflite                        # Предпочтительно (TensorFlow Lite)
    или
    signature_verification_model_fixed.h5  # Альтернатива (Keras)
    или
    signature_verification_model.h5      # Альтернатива (Keras)
    """)
