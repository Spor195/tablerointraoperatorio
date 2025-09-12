# app.py
import io
from datetime import datetime

import streamlit as st
from PIL import Image, ImageOps


# -----------------------------
# Configuración de la página
# -----------------------------
st.set_page_config(
    page_title="Convertidor a Escala de Grises",
    page_icon="🖤",
    layout="wide",
)

st.title("🖤 Convertidor de imágenes a escala de grises")
st.caption("Cargue una imagen, visualice el resultado y descárguelo en formato PNG.")


# -----------------------------
# Barra lateral (instrucciones)
# -----------------------------
with st.sidebar:
    st.header("Instrucciones")
    st.markdown(
        """
1. **Cargue** una imagen (JPG/PNG/TIFF/BMP/WebP).
2. **Revise** la vista previa.
3. **Convierta** automáticamente a escala de grises.
4. **Descargue** el resultado.
        """
    )
    st.divider()
    st.markdown("**Opciones**")
    keep_size = st.checkbox(
        "Mantener tamaño original", value=True,
        help="Si se desmarca, puede redimensionar la imagen de salida."
    )
    if not keep_size:
        width = st.number_input("Ancho (px)", min_value=16, max_value=8192, value=800, step=16)
    else:
        width = None

    dither = st.checkbox(
        "Usar difuminado (dithering)", value=False,
        help="Puede mejorar la percepción tonal en algunas imágenes."
    )


# -----------------------------
# Carga de imagen
# -----------------------------
uploaded = st.file_uploader(
    "Arrastre y suelte una imagen aquí o haga clic para seleccionar.",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
)

camera = st.camera_input("O capture una imagen con su cámara (opcional)")

# Selecciona la fuente efectiva
file_source = uploaded if uploaded is not None else camera

if not file_source:
    st.info("Esperando una imagen…")
    st.stop()

# Abre la imagen y corrige orientación EXIF
image = Image.open(file_source)
image = ImageOps.exif_transpose(image).convert("RGB")  # normaliza a RGB


# -----------------------------
# Procesamiento a escala de grises
# -----------------------------
# Conversión a 'L' (8 bits, 256 niveles de gris)
if dither:
    gray = ImageOps.grayscale(image)  # Pillow aplica difuminado por defecto en algunas rutas
else:
    # Conversión sin dithering explícito: evita patrones indeseados
    gray = image.convert("L", dither=Image.NONE)

# Redimensionamiento si corresponde
if width:
    w0, h0 = gray.size
    new_height = int(h0 * (width / w0))
    gray = gray.resize((width, new_height), resample=Image.BICUBIC)

# -----------------------------
# Visualización lado a lado
# -----------------------------
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Original")
    st.image(image, use_container_width=True)
    st.write(f"**Tamaño:** {image.size[0]}×{image.size[1]} px | **Modo:** RGB")

with col2:
    st.subheader("Escala de grises")
    st.image(gray, use_container_width=True, clamp=True)
    st.write(f"**Tamaño:** {gray.size[0]}×{gray.size[1]} px | **Modo:** L (8-bit)")

# -----------------------------
# Descarga
# -----------------------------
buf = io.BytesIO()
gray.save(buf, format="PNG")
buf.seek(0)

filename = f"grayscale_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
st.download_button(
    label="⬇️ Descargar imagen en gris (PNG)",
    data=buf,
    file_name=filename,
    mime="image/png",
)
