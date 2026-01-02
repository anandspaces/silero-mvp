import torch
from silero import silero_tts
from aksharamukha import transliterate
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load Indic model
model, _ = silero_tts(
    language="indic",
    speaker="v4_indic"
)

model.to(device)

# Original Hindi text
hindi_text = "Namastey, yeh silero ka safal parikshan hai, yeh dekstora ko nayi bulandiyon tak lekar jayega!"

# Romanize to ISO
roman_text = transliterate.process(
    "Devanagari",
    "ISO",
    hindi_text
)

print("Romanized:", roman_text)

audio = model.apply_tts(
    text=roman_text,
    speaker="hindi_male",  # or "hindi_female"
    sample_rate=48000
)

sf.write("hindi_output_roman.wav", audio, 48000)
print("Saved hindi_output.wav")
