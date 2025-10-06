import streamlit as st
import os

# Настройка страницы
st.set_page_config(
    page_title="DeepSign - Тест",
    page_icon="💳",
    layout="wide"
)

st.title("DeepSign - Тест запуска")
st.write("Если вы видите это сообщение, то Streamlit работает корректно!")

# Проверка наличия файлов
st.subheader("Проверка файлов:")
if os.path.exists('signature_verification_model_fixed.h5'):
    st.success("✅ Модель signature_verification_model_fixed.h5 найдена")
elif os.path.exists('signature_verification_model.h5'):
    st.success("✅ Модель signature_verification_model.h5 найдена")
else:
    st.error("❌ Модель не найдена!")

# Проверка зависимостей
st.subheader("Проверка зависимостей:")
try:
    import tensorflow as tf
    st.success(f"✅ TensorFlow {tf.__version__}")
except ImportError as e:
    st.error(f"❌ TensorFlow: {e}")

try:
    import cv2
    st.success(f"✅ OpenCV {cv2.__version__}")
except ImportError as e:
    st.error(f"❌ OpenCV: {e}")

try:
    import numpy as np
    st.success(f"✅ NumPy {np.__version__}")
except ImportError as e:
    st.error(f"❌ NumPy: {e}")

try:
    from PIL import Image
    st.success("✅ Pillow")
except ImportError as e:
    st.error(f"❌ Pillow: {e}")

st.write("Тест завершен!")

