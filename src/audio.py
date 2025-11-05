import librosa
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio
import numpy as np

# Load clean conversation data
clean_speech, sr = librosa.load(librosa.example(''), sr=None)

# some background noise data 
np.random.seed(42)
background_noise = np.random.normal(0, 0.03, clean_speech.shape)

# noisy conversation
noisy_conversation = clean_speech + background_noise
sf.write("noisy_conversation.wav", noisy_conversation, sr)

print("noisy audio:")
display(Audio("noisy_conversation.wav"))

#  waveforms
plt.figure(figsize=(12, 4))
plt.plot(noisy_conversation, label="Noisy conversation", alpha=0.7)
plt.plot(clean_speech, label="Clean conversation", alpha=0.6)
plt.title("Noisy vs Clean Waveform")
plt.legend()
plt.show()

# background noise  
noise_sample = noisy_conversation[:int(0.5 * sr)]

# noise reduction
denoised_conversation = nr.reduce_noise(y=noisy_conversation, y_noise=noise_sample, sr=sr)

# denoised output
sf.write("denoised_conversation.wav", denoised_conversation, sr)

print("Denoised conversation:")
display(Audio("denoised_conversation.wav"))

#Visual  
plt.figure(figsize=(12, 4))
plt.plot(noisy_conversation, label="Noisy", alpha=0.5)
plt.plot(denoised_conversation, label="Denoised", alpha=0.8)
plt.title("Before and After Noise Reduction")
plt.legend()
plt.show()
