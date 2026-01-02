import torch
from silero import silero_tts
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

model, _ = silero_tts(
    language="ru",
    speaker="v5_ru"
)

model.to(device)

audio = model.apply_tts(
    text="Привет! Это успешный тест Silero text to speech.",
    speaker="xenia",
    sample_rate=48000
)

sf.write("russian_output.wav", audio, 48000)
print("Saved russian_output.wav")
