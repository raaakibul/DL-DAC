from speechbrain.pretrained import SpectralMaskEnhancement
import torchaudio
import torch
import matplotlib.pyplot as plt
#  pre-trained model
enhancer = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="pretrained_model"
)

# noisy conversation audio
noisy_file = "noisy_conversation.wav"
noisy_waveform, sr = torchaudio.load(noisy_file)

# denoise the audio using deep learning
enhanced = enhancer.enhance_batch(noisy_waveform)

# output
torchaudio.save("enhanced_conversation.wav", enhanced.cpu(), sr)

print("Deep Learning):")
display(Audio("enhanced_conversation.wav"))

#  comparison
noisy_np = noisy_waveform[0].numpy()
enhanced_np = enhanced[0].detach().cpu().numpy()

plt.figure(figsize=(12, 4))
plt.plot(noisy_np, label="Noisy", alpha=0.5)
plt.plot(enhanced_np, label="Enhanced", alpha=0.8)
plt.legend()
plt.title("Deep Learning Noise Reduction Comparison")
plt.show()