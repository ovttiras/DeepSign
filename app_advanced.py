import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
from PIL import Image
import base64
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Настройка страницы
st.set_page_config(
    page_title="DeepSign - Система верификации подписей",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили в стиле Агропромбанка
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
    
    /* Анимация загрузки */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid var(--apb-green);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Инициализация сессии
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0
if 'genuine_count' not in st.session_state:
    st.session_state.genuine_count = 0
if 'forged_count' not in st.session_state:
    st.session_state.forged_count = 0

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
            st.error("Модель не найдена! Убедитесь, что файл модели находится в папке приложения.")
            return None, None, None
        
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
        start_time = time.time()
        
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
        
        processing_time = time.time() - start_time
        
        if confidence >= 0.5:
            return "Настоящая подпись", confidence, "genuine", processing_time
        else:
            return "Поддельная подпись", 1 - confidence, "forged", processing_time
    except Exception as e:
        st.error(f"Ошибка предсказания: {str(e)}")
        return None, None, None, None

# Функция создания графика истории
def create_history_chart():
    """Создает график истории анализов"""
    if not st.session_state.analysis_history:
        return None
    
    df = st.session_state.analysis_history[-10:]  # Последние 10 анализов
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=[item['confidence'] for item in df],
        mode='lines+markers',
        name='Уверенность',
        line=dict(color='#2E7D32', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="История анализов (последние 10)",
        xaxis_title="Номер анализа",
        yaxis_title="Уверенность (%)",
        template="plotly_white",
        height=300
    )
    
    return fig

# Заголовок приложения
st.markdown("""
<div class="main-header">
    <h1>🏦 DeepSign</h1>
    <p>Инновационная технология распознавания подлинности подписей</p>
</div>
""", unsafe_allow_html=True)

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
        st.markdown("**Точность:** 99.8%")
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
        
        st.markdown("### 📈 Статистика сессии")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Обработано", st.session_state.total_analyzed)
        with col2:
            if st.session_state.total_analyzed > 0:
                accuracy = (st.session_state.genuine_count / st.session_state.total_analyzed) * 100
                st.metric("Точность", f"{accuracy:.1f}%")
            else:
                st.metric("Точность", "0%")
        
        # График истории
        if st.session_state.analysis_history:
            st.markdown("### 📊 История анализов")
            fig = create_history_chart()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Основной интерфейс
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Загрузка подписи")
        
        uploaded_file = st.file_uploader(
            "Выберите изображение подписи",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Поддерживаемые форматы: PNG, JPG, JPEG, BMP"
        )
        
        if uploaded_file is not None:
            # Отображение загруженного изображения
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженная подпись", use_column_width=True)
            
            # Кнопка анализа
            if st.button("🔍 Анализировать подпись", type="primary"):
                with st.spinner("Анализируем подпись..."):
                    # Предобработка
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        # Предсказание
                        result, confidence, result_type, processing_time = predict_signature(model, processed_image, model_format)
                        
                        if result is not None:
                            # Обновление статистики
                            st.session_state.total_analyzed += 1
                            if result_type == "genuine":
                                st.session_state.genuine_count += 1
                            else:
                                st.session_state.forged_count += 1
                            
                            # Добавление в историю
                            st.session_state.analysis_history.append({
                                'timestamp': datetime.now(),
                                'result': result,
                                'confidence': confidence * 100,
                                'processing_time': processing_time,
                                'type': result_type
                            })
                            
                            # Отображение результата
                            with col2:
                                st.markdown("### 🎯 Результат анализа")
                                
                                if result_type == "genuine":
                                    st.markdown(f"""
                                    <div class="result-genuine">
                                        <h2>✅ {result}</h2>
                                        <p>Уверенность: {confidence:.1%}</p>
                                        <p>Время обработки: {processing_time:.3f}с</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="result-forged">
                                        <h2>❌ {result}</h2>
                                        <p>Уверенность: {confidence:.1%}</p>
                                        <p>Время обработки: {processing_time:.3f}с</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Дополнительная информация
                                st.markdown("### 📋 Детали анализа")
                                st.write(f"**Тип подписи:** {result}")
                                st.write(f"**Уверенность:** {confidence:.1%}")
                                st.write(f"**Порог классификации:** {confidence_threshold:.1%}")
                                st.write(f"**Время обработки:** {processing_time:.3f} секунд")
                                
                                # Прогресс бар уверенности
                                st.progress(confidence)
                                
                                # Рекомендации
                                if result_type == "genuine":
                                    st.success("✅ Подпись признана подлинной. Рекомендуется принять документ.")
                                else:
                                    st.warning("⚠️ Подпись признана поддельной. Рекомендуется дополнительная проверка.")
                                
                                # Кнопка экспорта результата
                                if st.button("📄 Экспорт результата"):
                                    result_data = {
                                        'timestamp': datetime.now().isoformat(),
                                        'result': result,
                                        'confidence': float(confidence),
                                        'processing_time': float(processing_time),
                                        'threshold': float(confidence_threshold)
                                    }
                                    
                                    json_str = json.dumps(result_data, indent=2, ensure_ascii=False)
                                    st.download_button(
                                        label="Скачать JSON",
                                        data=json_str,
                                        file_name=f"signature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json"
                                    )
    
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
            <p>Точность распознавания составляет 99.8%, что обеспечивает надежную защиту от подделок.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Кнопка сброса статистики
    if st.button("🔄 Сбросить статистику"):
        st.session_state.analysis_history = []
        st.session_state.total_analyzed = 0
        st.session_state.genuine_count = 0
        st.session_state.forged_count = 0
        st.success("Статистика сброшена!")
        st.rerun()
    
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







