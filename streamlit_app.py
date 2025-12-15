import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# Asetetaan sivun konfiguraatio
st.set_page_config(page_title="Liikuntadatan Analyysi", layout="wide")
st.title("Liikuntadatan Analyysi")

# Lataa data
try:
    acceleration_data = pd.read_csv("Data/LoppuKiihtyvyys.csv")
    gps_data = pd.read_csv("Data/LoppuPaikannus.csv")
except FileNotFoundError:
    # Jos paikallista tiedostoa ei löydy, kokeillaan GitHub-repositoriota
    try:
        repo_url = "https://raw.githubusercontent.com/Hesseli/FysiikanLoppuprojekti/main/"
        acceleration_data = pd.read_csv(repo_url + "Data/LoppuKiihtyvyys.csv")
        gps_data = pd.read_csv(repo_url + "Data/LoppuPaikannus.csv")
    except:
        st.error("Data-tiedostoja ei löydy. Varmista, että Data-hakemisto on oikein sijoitettu.")
        st.stop()

# Puhdistetaan GPS-data
gps_data_clean = gps_data.dropna(subset=['Latitude (°)', 'Longitude (°)', 'Velocity (m/s)'])

# 1. ASKELMÄÄRÄN LASKENTA SUODATUKSELLA
st.header("Askelmäärän määritys")

# Lasketaan kiihtyvyyden suuruus (magnitude)
acceleration_data['Magnitude'] = np.sqrt(
    acceleration_data['Linear Acceleration x (m/s^2)']**2 +
    acceleration_data['Linear Acceleration y (m/s^2)']**2 +
    acceleration_data['Linear Acceleration z (m/s^2)']**2
)

# Näytteenottotaajuus (oletuksena 50 Hz, mutta lasketaan datasta)
sampling_rate = 1 / (acceleration_data['Time (s)'].iloc[1] - acceleration_data['Time (s)'].iloc[0])

# Butterworth-suodatin (alipäästösuodatin askelten tunnistamiseen)
cutoff_freq = 3  # Hz
order = 4
normalized_cutoff = cutoff_freq / (sampling_rate / 2)
b, a = signal.butter(order, normalized_cutoff, btype='low')
filtered_magnitude = signal.filtfilt(b, a, acceleration_data['Magnitude'])

# Etsitään huiput (askeleet)
threshold = np.mean(filtered_magnitude) + 0.5 * np.std(filtered_magnitude)
peaks, _ = signal.find_peaks(filtered_magnitude, height=threshold, distance=int(sampling_rate * 0.3))
step_count_filtered = len(peaks)

# 2. ASKELMÄÄRÄ FOURIER-ANALYYSIN PERUSTEELLA
# Valitaan z-komponentti pystysuuntaiselle kiihtyvyydelle
z_acceleration = acceleration_data['Linear Acceleration z (m/s^2)'].values
z_acceleration_detrended = signal.detrend(z_acceleration)

# Fourier-muunnos
fft_values = np.abs(fft(z_acceleration_detrended))
fft_freqs = fftfreq(len(z_acceleration_detrended), 1/sampling_rate)

# Positiiviset taajuudet
positive_freqs_idx = fft_freqs > 0
positive_freqs = fft_freqs[positive_freqs_idx]
positive_fft = fft_values[positive_freqs_idx]

# Etsitään dominantti taajuus askelille (tyypillisesti 1-3 Hz)
step_freq_range = (positive_freqs >= 1) & (positive_freqs <= 3)
step_freq_idx = np.argmax(positive_fft[step_freq_range])
dominant_step_freq = positive_freqs[step_freq_range][step_freq_idx]
step_count_fft = int(dominant_step_freq * acceleration_data['Time (s)'].iloc[-1])

# 3. KESKINOPEUS JA MATKA (GPS-datasta)
velocities = gps_data_clean['Velocity (m/s)'].values
time_gps = gps_data_clean['Time (s)'].values

average_velocity = np.mean(velocities[velocities > 0])  # Poista nollanopeudet

# Lasketaan matka integroimalla nopeus
distances = np.diff(time_gps) * velocities[:-1]
total_distance = np.sum(distances)

# 4. ASKELPITUUS
step_length = total_distance / max(step_count_filtered, 1)

# Näytetään tulokset
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Askelmäärä (suodatus)", step_count_filtered)

with col2:
    st.metric("Askelmäärä (Fourier)", step_count_fft)

with col3:
    st.metric("Keskinopeus (GPS)", f"{average_velocity:.2f} m/s")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Kuljettu matka", f"{total_distance:.2f} m")

with col5:
    st.metric("Askelpituus", f"{step_length:.2f} m")

with col6:
    st.metric("Dominantti askelitaajuus", f"{dominant_step_freq:.2f} Hz")

# 5. VISUALISOINNIT
st.header("Visualisoinnit")

# Kuvaaja 1: Suodatettu kiihtyvyysdata
st.subheader("Suodatettu kiihtyvyysdata")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(acceleration_data['Time (s)'], filtered_magnitude, label='Suodatettu suuruus', linewidth=1.5)
ax1.scatter(acceleration_data['Time (s)'].iloc[peaks], filtered_magnitude[peaks], 
            color='red', s=30, label=f'Askeleet (n={step_count_filtered})', zorder=5)
ax1.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label='Kynnys')
ax1.set_xlabel('Aika (s)')
ax1.set_ylabel('Kiihtyvyyden suuruus (m/s²)')
ax1.set_title('Suodatettu kiihtyvyysdata ja tunnistetut askeleet')
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# Kuvaaja 2: Tehospektritiheys (Power Spectral Density)
st.subheader("Tehospektritiheys (z-akseli)")
fig2, ax2 = plt.subplots(figsize=(12, 4))
frequencies, psd = signal.welch(z_acceleration_detrended, fs=sampling_rate, nperseg=4096)
ax2.semilogy(frequencies, psd, linewidth=1.5)
ax2.axvline(x=dominant_step_freq, color='red', linestyle='--', 
            label=f'Dominantti taajuus: {dominant_step_freq:.2f} Hz')
ax2.set_xlabel('Taajuus (Hz)')
ax2.set_ylabel('Tehospektritiheys')
ax2.set_title('Tehospektritiheys')
ax2.set_xlim(0, 10)
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# Kuvaaja 3: Reitti kartalla
st.subheader("Reitti kartalla")

if len(gps_data_clean) > 0:
    center_lat = gps_data_clean['Latitude (°)'].mean()
    center_lon = gps_data_clean['Longitude (°)'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=16,
        tiles='OpenStreetMap'
    )
    
    # Piirretään reittiviiva
    route_points = list(zip(gps_data_clean['Latitude (°)'], gps_data_clean['Longitude (°)']))
    folium.PolyLine(
        route_points,
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(m)
    
    # Lisätään alku- ja loppupisteet
    folium.Marker(
        location=[gps_data_clean['Latitude (°)'].iloc[0], gps_data_clean['Longitude (°)'].iloc[0]],
        popup='Lähtöpiste',
        icon=folium.Icon(color='green', prefix='fa', icon='circle')
    ).add_to(m)
    
    folium.Marker(
        location=[gps_data_clean['Latitude (°)'].iloc[-1], gps_data_clean['Longitude (°)'].iloc[-1]],
        popup='Päätepiste',
        icon=folium.Icon(color='red', prefix='fa', icon='circle')
    ).add_to(m)
    
    st_folium(m, width=1200, height=600)
else:
    st.warning("GPS-dataa ei ole saatavilla.")
