import streamlit as st
import numpy as np
from PIL import Image
from scipy.io.wavfile import write
import trimesh
import plotly.graph_objects as go
import os
import glob

st.set_page_config(page_title="NASA Earth Information Jukebox", page_icon="🪐", layout="wide")

SCALES = {
    "minor": [110.00, 130.81, 146.83, 164.81, 196.00, 220.00, 261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 659.25],
    "major": [130.81, 146.83, 164.81, 196.00, 220.00, 261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25],
    "dramatic": [110.00, 116.54, 138.59, 146.83, 164.81, 174.61, 207.65, 220.00, 233.08, 277.18, 293.66, 329.63, 440.00]
}


def generate_space_pad(freq, duration, wave_type="soft", sample_rate=44100, amplitude=0.6):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave1 = np.sin(2 * np.pi * freq * t)
    wave2 = 0.5 * np.sin(2 * np.pi * (freq * 1.004) * t)
    wave3 = 0.3 * np.sin(2 * np.pi * (freq * 0.5) * t)
    wave = (wave1 + wave2 + wave3 + 0.25 * np.sin(4 * np.pi * freq * t)) if wave_type == "sharp" else (
                wave1 + wave2 + wave3)

    attack = int(sample_rate * (duration * 0.3))
    release = int(sample_rate * (duration * 0.4))
    sustain = len(t) - attack - release
    envelope = np.concatenate([np.linspace(0, 1, attack), np.ones(max(0, sustain)), np.linspace(1, 0, release)])[
        :len(t)]
    return wave * envelope * amplitude


def apply_space_reverb(stereo_audio, sample_rate=44100):
    output = np.copy(stereo_audio)
    for delay_sec, decay in [(0.18, 0.45), (0.35, 0.25), (0.55, 0.15)]:
        delay_samples = int(sample_rate * delay_sec)
        if delay_samples < len(output):
            output[delay_samples:, 0] += stereo_audio[:-delay_samples, 1] * decay
            output[delay_samples:, 1] += stereo_audio[:-delay_samples, 0] * decay
    return output


def model_3d_to_spatial_music(glb_path, output_wav, total_duration=9.0):
    # Çalışan kesin çözüm: force='mesh'
    mesh = trimesh.load(glb_path, force='mesh')
    vertices = np.array(mesh.vertices, dtype=np.float32)

    v_min, v_max = vertices.min(axis=0), vertices.max(axis=0)
    v_range = np.where((v_max - v_min) == 0, 1.0, v_max - v_min)
    norm_v = (vertices - v_min) / v_range

    steps = 36
    active_scale = SCALES["minor"]
    sample_rate = 44100
    total_samples = int(sample_rate * total_duration)
    step_samples = total_samples // steps
    stereo_audio = np.zeros((total_samples, 2), dtype=np.float32)

    t_full = np.linspace(0, total_duration, total_samples, False)
    drone = 0.25 * np.sin(2 * np.pi * (min(active_scale) * 0.5) * t_full)
    stereo_audio[:, 0] += drone
    stereo_audio[:, 1] += drone

    for x_step in range(steps):
        mask = (norm_v[:, 0] >= x_step / steps) & (norm_v[:, 0] < (x_step + 1) / steps)
        slice_points = norm_v[mask]
        if len(slice_points) > 0:
            peak_y = np.max(slice_points[:, 1])
            note_idx = int(np.clip(peak_y, 0, 1) * (len(active_scale) - 1))
            freq = active_scale[note_idx]
            amp = np.clip(0.4 + (np.mean(slice_points[:, 2]) * 0.55), 0.4, 0.95)
            mono_note = generate_space_pad(freq, (total_duration / steps) * 1.8, "soft", sample_rate, amp)

            pan = x_step / (steps - 1)
            start_idx = x_step * step_samples
            end_idx = min(start_idx + len(mono_note), total_samples)
            valid_len = end_idx - start_idx
            stereo_audio[start_idx:end_idx, 0] += mono_note[:valid_len] * np.cos(pan * np.pi / 2)
            stereo_audio[start_idx:end_idx, 1] += mono_note[:valid_len] * np.sin(pan * np.pi / 2)

    stereo_audio = apply_space_reverb(stereo_audio, sample_rate)
    stereo_audio = np.nan_to_num(stereo_audio, nan=0.0)
    max_val = np.max(np.abs(stereo_audio))
    if max_val > 0:
        stereo_audio = (stereo_audio / max_val) * 0.95
    write(output_wav, sample_rate, np.int16(stereo_audio * 32767))
    return vertices


def image_2d_to_spatial_music(image_path, output_wav, total_duration=9.0):
    img = Image.open(image_path).convert("RGB")
    steps = 32
    data = np.array(img.resize((steps, 32)), dtype=np.float32)

    avg_r, avg_g, avg_b = np.mean(data[:, :, 0]), np.mean(data[:, :, 1]), np.mean(data[:, :, 2])
    if avg_r > avg_b and avg_r > avg_g:
        active_scale, wave_type, mood = SCALES[
            "dramatic"], "sharp", "Sıcaklık / Kuraklık Atmosferi (Dramatik Hicaz Gamı)"
    elif avg_b > avg_r and avg_b > avg_g:
        active_scale, wave_type, mood = SCALES["minor"], "soft", "Su / Buzul / Okyanus Atmosferi (Derin Minör Gamı)"
    else:
        active_scale, wave_type, mood = SCALES["major"], "soft", "Biyosfer / Orman Atmosferi (Parlak Majör Gamı)"

    sample_rate = 44100
    total_samples = int(sample_rate * total_duration)
    step_samples = total_samples // steps
    stereo_audio = np.zeros((total_samples, 2), dtype=np.float32)

    t_full = np.linspace(0, total_duration, total_samples, False)
    drone = 0.25 * np.sin(2 * np.pi * (min(active_scale) * 0.5) * t_full)
    stereo_audio[:, 0] += drone
    stereo_audio[:, 1] += drone

    for x in range(steps):
        column = data[:, x, :]
        brightness_col = np.mean(column, axis=1)
        peak_y = np.argmax(brightness_col)
        note_idx = int(((31 - peak_y) / 31.0) * (len(active_scale) - 1))
        freq = active_scale[note_idx]
        amp = np.clip(0.4 + (np.std(brightness_col) / 128.0), 0.4, 0.95)
        mono_note = generate_space_pad(freq, (total_duration / steps) * 1.8, wave_type, sample_rate, amp)

        pan = x / (steps - 1)
        start_idx = x * step_samples
        end_idx = min(start_idx + len(mono_note), total_samples)
        valid_len = end_idx - start_idx
        stereo_audio[start_idx:end_idx, 0] += mono_note[:valid_len] * np.cos(pan * np.pi / 2)
        stereo_audio[start_idx:end_idx, 1] += mono_note[:valid_len] * np.sin(pan * np.pi / 2)

    stereo_audio = apply_space_reverb(stereo_audio, sample_rate)
    stereo_audio = np.nan_to_num(stereo_audio, nan=0.0)
    max_val = np.max(np.abs(stereo_audio))
    if max_val > 0:
        stereo_audio = (stereo_audio / max_val) * 0.95
    write(output_wav, sample_rate, np.int16(stereo_audio * 32767))
    return mood


# --- WEB ARAYÜZÜ ---
st.title("🪐 NASA Earth Information Jukebox")
st.markdown(
    "**NASA Space Apps Challenge 2026** | *Uydu Verileri ve 3D Gezegen Modelleri İçin Uzaysal Seslendirme Sistemi*")
st.divider()

yerel_dosyalar = glob.glob("*.glb") + glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")

if yerel_dosyalar:
    secilen_dosya = st.sidebar.selectbox("İşlenecek Dosyayı Seçin:", yerel_dosyalar)
    sure = st.sidebar.slider("Müzik Süresi (Saniye)", 5, 20, 9)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌍 Veri Önizleme")
        if secilen_dosya.lower().endswith(".glb"):
            mesh = trimesh.load(secilen_dosya, force='mesh')
            verts = np.array(mesh.vertices)
            sample_verts = verts[::max(1, len(verts) // 3000)]
            fig = go.Figure(data=[go.Scatter3d(
                x=sample_verts[:, 0], y=sample_verts[:, 1], z=sample_verts[:, 2],
                mode='markers', marker=dict(size=2.5, color=sample_verts[:, 1], colorscale='Viridis')
            )])
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=350)
            st.plotly_chart(fig)
            st.caption("3D Modeli fareyle tutup döndürebilirsiniz.")
        else:
            st.image(secilen_dosya)

    with col2:
        st.subheader("🎧 Uzaysal Müzik Kutusu")
        if st.button("🚀 Veriyi Uzaysal Müziğe Dönüştür", type="primary"):
            with st.spinner("Uzaysal ses üretiliyor..."):
                cikti_wav = "web_uzaysal_ses.wav"
                if secilen_dosya.lower().endswith(".glb"):
                    model_3d_to_spatial_music(secilen_dosya, cikti_wav, float(sure))
                    st.success("3D Uzaysal Topografya Sentezi Tamamlandı!")
                else:
                    mood = image_2d_to_spatial_music(secilen_dosya, cikti_wav, float(sure))
                    st.success(f"Atmosfer Analizi: {mood}")

                # Tarayıcı oynatıcısını ekrana getir
                with open(cikti_wav, "rb") as f:
                    st.audio(f.read(), format="audio/wav")
                st.info("👆 Dinlemek için yukarıdaki oynatıcı çubuğunun solundaki **▶ (Play)** simgesine tıklayın.")
else:
    st.warning("Klasörde .glb, .jpg veya .png dosyası bulunamadı.")
