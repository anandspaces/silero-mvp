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
hindi_text = "नमस्ते, यह सिलेरो का सफल परीक्षण है, यह डेक्सटोरा को नई बुलंदियों तक लेकर जाएगा।"

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

sf.write("hindi_output.wav", audio, 48000)
print("Saved hindi_output.wav")
