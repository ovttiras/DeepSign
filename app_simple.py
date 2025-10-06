import streamlit as st
import os

# Настройка страницы
st.set_page_config(
    page_title="DeepSign - Простая версия",
    page_icon="💳",
    layout="wide"
)

# Заголовок
st.title("💳 DeepSign")
st.subheader("Система верификации подписей")

# Информация о системе
st.info("Это упрощенная версия DeepSign для тестирования")

# Проверка зависимостей
st.subheader("🔍 Проверка системы")

col1, col2 = st.columns(2)

with col1:
    st.write("**Python модули:**")
    try:
        import streamlit as st_module
        st.success(f"✅ Streamlit {st_module.__version__}")
    except:
        st.error("❌ Streamlit")
    
    try:
        import tensorflow as tf
        st.success(f"✅ TensorFlow {tf.__version__}")
    except:
        st.error("❌ TensorFlow")
    
    try:
        import cv2
        st.success(f"✅ OpenCV {cv2.__version__}")
    except:
        st.error("❌ OpenCV")

with col2:
    st.write("**Файлы модели:**")
    if os.path.exists('model.tflite'):
        st.success("✅ model.tflite (TensorFlow Lite)")
    elif os.path.exists('signature_verification_model_fixed.h5'):
        st.success("✅ signature_verification_model_fixed.h5")
    elif os.path.exists('signature_verification_model.h5'):
        st.success("✅ signature_verification_model.h5")
    else:
        st.warning("⚠️ Модель не найдена")

# Простой интерфейс
st.subheader("📤 Загрузка изображения")

uploaded_file = st.file_uploader(
    "Выберите изображение подписи",
    type=['png', 'jpg', 'jpeg', 'bmp']
)

if uploaded_file is not None:
    from PIL import Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)
    
    if st.button("🔍 Анализировать"):
        st.info("⚠️ Для полного анализа необходимо обучить модель")
        st.write("Запустите Signature_Verification.ipynb для обучения модели")

# Инструкции
st.subheader("📋 Инструкции по запуску")

st.markdown("""
### Для полного запуска:

1. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Обучите модель:**
   - Откройте `Signature_Verification.ipynb`
   - Выполните все ячейки
   - Сохраните модель

3. **Запустите приложение:**
   ```bash
   streamlit run app.py
   ```

### Альтернативные команды:
- `streamlit run app_advanced.py` - расширенная версия
- `run_app.bat` - автоматический запуск
- `diagnose.bat` - диагностика проблем
""")

# Футер
st.markdown("---")
st.markdown("© 2025 DeepSign was developed by Oleg Tinkov - Система верификации подписей")
