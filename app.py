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
    initial_sidebar_state="expanded"
)

# CSS стили в стиле Тинькофф
st.markdown("""
<style>
    /* Основные цвета АПБ */
    :root {
        --apb-green: #2E7D32;
        --apb-dark-green: #1B5E20;
        --apb-light-green: #4CAF50;
        --apb-gold: #FFC107;
        --apb-white: #FFFFFF;
        --apb-gray: #F5F5F5;
        --apb-dark-gray: #424242;
    }
    
    /* Главный контейнер */
    .main-header {
        background: linear-gradient(135deg, var(--apb-green) 0%, var(--apb-dark-green) 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Карточки */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--apb-green);
    }
    
    .card h3 {
        color: var(--apb-green);
        margin-top: 0;
    }
    
    /* Кнопки */
    .stButton > button {
        background: linear-gradient(135deg, var(--apb-green) 0%, var(--apb-dark-green) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Результаты */
    .result-genuine {
        background: linear-gradient(135deg, #4CAF50, #2E7D32);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-forged {
        background: linear-gradient(135deg, #F44336, #D32F2F);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Загрузка файла */
    .upload-section {
        background: var(--apb-gray);
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed var(--apb-green);
        text-align: center;
    }
    
    /* Статистика */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: var(--apb-green);
    }
    
    .stat-label {
        color: var(--apb-dark-gray);
        font-size: 0.9rem;
    }
    
    /* Сайдбар */
    .sidebar .sidebar-content {
        background: var(--apb-gray);
    }
    
    /* Прогресс бар */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--apb-green), var(--apb-light-green));
    }
</style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown("""
<div class="main-header">
    <h1>💳 DeepSign</h1>
    <p>Инновационная технология распознавания подлинности подписей</p>
</div>
""", unsafe_allow_html=True)

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
        
        # Нормализация
        img_array = img_array / 255.0
        
        # Добавляем канал
        img_array = img_array.reshape(1, 155, 220, 1)
        
        return img_array
    except Exception as e:
        st.error(f"Ошибка предобработки изображения: {str(e)}")
        return None

# Функция предсказания
def predict_signature(model, image, model_format="keras"):
    """Предсказание подлинности подписи"""
    try:
        if model_format == "tflite":
            # Предсказание для TensorFlow Lite модели
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            # Устанавливаем входные данные
            model.set_tensor(input_details[0]['index'], image.astype(np.float32))
            
            # Запускаем инференс
            model.invoke()
            
            # Получаем результат
            prediction = model.get_tensor(output_details[0]['index'])
            confidence = prediction[0][0]
        else:
            # Предсказание для Keras модели
            prediction = model.predict(image, verbose=0)
            confidence = prediction[0][0]
        
        if confidence >= 0.5:
            return "Настоящая подпись", confidence, "genuine"
        else:
            return "Поддельная подпись", 1 - confidence, "forged"
    except Exception as e:
        st.error(f"Ошибка предсказания: {str(e)}")
        return None, None, None

# Загрузка модели
model, model_type, model_format = load_signature_model()

if model is not None:
    st.success(f"✅ Модель загружена: {model_type}")
    if model_format == "tflite":
        st.info("🚀 Используется оптимизированная TensorFlow Lite модель")
    
    # Сайдбар с информацией
    with st.sidebar:
        st.markdown("### 📊 Информация о системе")
        st.markdown(f"**Тип модели:** {model_type}")
        st.markdown("**Точность на обучающей выборке:** 99.9%")
        st.markdown("**Точность на тестовой выборке:** 79%")
        st.markdown("**Время обработки:** < 1 сек")
        
        st.markdown("### 🔧 Настройки")
        confidence_threshold = st.slider(
            "Порог уверенности", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.1,
            help="Пороговое значение для классификации"
        )
        
        st.markdown("### 📈 Статистика")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Обработано", "0", "0")
        with col2:
            st.metric("Точность (тест)", "79%", "0.1%")
    
    # Основной интерфейс
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Примеры файлов с подписями")
        
        # Примеры файлов
        example_files = [
            ("example/original signature_1.png", "Настоящая подпись 1"),
            ("example/original signature_2.png", "Настоящая подпись 2"),
            ("example/fake signature_1.png", "Поддельная подпись 1"),
            ("example/fake signature_2.png", "Поддельная подпись 2")
        ]
        
        for file_path, description in example_files:
            if os.path.exists(file_path):
                st.markdown(f"• **{description}:** `{file_path}`")
            else:
                st.markdown(f"• **{description}:** `{file_path}` (файл не найден)")
        
        st.markdown("---")
        st.markdown("### 📤 Загрузка подписи")
        
        uploaded_file = st.file_uploader(
            "Выберите изображение подписи",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Поддерживаемые форматы: PNG, JPG, JPEG, BMP"
        )
        
        if uploaded_file is not None:
            # Отображение загруженного изображения
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженная подпись", use_container_width=True)
            
            # Кнопка анализа
            if st.button("🔍 Анализировать подпись", type="primary"):
                with st.spinner("Анализируем подпись..."):
                    # Предобработка
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        # Предсказание
                        result, confidence, result_type = predict_signature(model, processed_image, model_format)
                        
                        if result is not None:
                            # Отображение результата
                            with col2:
                                st.markdown("### 🎯 Результат анализа")
                                
                                if result_type == "genuine":
                                    st.markdown(f"""
                                    <div class="result-genuine">
                                        <h2>✅ {result}</h2>
                                        <p>Уверенность: {confidence:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="result-forged">
                                        <h2>❌ {result}</h2>
                                        <p>Уверенность: {confidence:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Дополнительная информация
                                st.markdown("### 📋 Детали анализа")
                                st.write(f"**Тип подписи:** {result}")
                                st.write(f"**Уверенность:** {confidence:.1%}")
                                st.write(f"**Порог классификации:** {confidence_threshold:.1%}")
                                
                                # Прогресс бар уверенности
                                st.progress(float(confidence))
                                
                                # Рекомендации
                                if result_type == "genuine":
                                    st.success("✅ Подпись признана подлинной. Рекомендуется принять документ.")
                                else:
                                    st.warning("⚠️ Подпись признана поддельной. Рекомендуется дополнительная проверка.")
    
    # Информационная секция
    st.markdown("---")
    st.markdown("### ℹ️ О системе верификации подписей")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🔬 Технология</h3>
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
    
    # Детальная информация о модели и архитектуре
    st.markdown("---")
    st.markdown("### 🧠 Архитектура нейронной сети")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🏗️ Архитектура модели</h3>
            <p><strong>Тип:</strong> Сверточная нейронная сеть (CNN)</p>
            <p><strong>Входной размер:</strong> 220×155×1 (градации серого)</p>
            <p><strong>Слои:</strong></p>
            <ul>
                <li>Conv2D + MaxPooling (извлечение признаков)</li>
                <li>Dropout (регуляризация)</li>
                <li>Dense слои (классификация)</li>
                <li>Sigmoid активация (бинарная классификация)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>📊 База данных CEDAR-Dataset</h3>
            <p><strong>Источник:</strong> CEDAR (Center of Excellence for Document Analysis and Recognition)</p>
            <p><strong>Ссылка на датасет:</strong> <a href="https://www.kaggle.com/datasets/shreelakshmigp/cedardataset/data" target="_blank">https://www.kaggle.com/datasets/shreelakshmigp/cedardataset/data</a></p>
            <p><strong>Размер:</strong> 55 авторов, 24 подписи каждого</p>
            <p><strong>Общее количество подписей:</strong> 2,640</p>
            <p><strong>Распределение:</strong> Равномерно представлены двумя классами - настоящие и поддельные подписи</p>
            <p><strong>Формат:</strong> Отсканированные изображения высокого качества</p>
            <p><strong>Особенности:</strong> Различные стили подписей, естественные вариации</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Дополнительная техническая информация
    st.markdown("### 🔧 Технические характеристики")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🎯 Предобработка</h3>
            <ul>
                <li>Изменение размера до 220×155</li>
                <li>Конвертация в градации серого</li>
                <li>Бинаризация изображения</li>
                <li>Нормализация пикселей [0,1]</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>⚙️ Обучение</h3>
            <ul>
                <li>Оптимизатор: Adam</li>
                <li>Функция потерь: Binary Crossentropy</li>
                <li>Метрики: Accuracy, Precision, Recall</li>
                <li>Валидация: 20% от датасета</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3>📈 Производительность</h3>
            <ul>
                <li>Точность (обучение): 99.9%</li>
                <li>Точность (тест): 79%</li>
                <li>Время инференса: < 1 сек</li>
                <li>Размер модели: ~2.5 МБ</li>
                <li>Совместимость: TensorFlow/Keras</li>
            </ul>
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

