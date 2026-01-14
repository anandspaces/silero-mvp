"""
Production-grade text normalization for multilingual TTS
Handles numbers, units, punctuation, and special cases
"""

import re
from typing import Dict, Callable
from num2words import num2words

class TTSTextNormalizer:
    """Normalize text for TTS across multiple languages"""
    
    # Language-specific number conversion
    LANG_MAP = {
        'en': 'en',
        'hi': 'hi',
        'ta': 'te',  # num2words uses 'te' for Tamil
        'te': 'te',
        'bn': 'bn',
        'gu': 'gu',
        'kn': 'kn',
        'ml': 'ml',
        'mni': 'en',  # Manipuri fallback to English
        'raj': 'hi',  # Rajasthani fallback to Hindi
    }
    
    def __init__(self):
        """Initialize normalizer with language-specific patterns"""
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for efficiency"""
        return {
            # Decimals (must come before integers)
            'decimal': re.compile(r'\b(\d+)\.(\d+)\b'),
            
            # Percentages
            'percentage': re.compile(r'\b(\d+(?:\.\d+)?)\s*%'),
            
            # Currency (common symbols)
            'currency_rs': re.compile(r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)'),
            'currency_dollar': re.compile(r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)'),
            
            # Time formats
            'time_colon': re.compile(r'\b(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)?\b'),
            'time_hhmm': re.compile(r'\b(\d{1,2})(\d{2})\s*hrs?\b'),
            
            # Dates
            'date_slash': re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b'),
            'date_dash': re.compile(r'\b(\d{1,2})-(\d{1,2})-(\d{2,4})\b'),
            
            # Ordinals (1st, 2nd, 3rd, etc)
            'ordinal': re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE),
            
            # Phone numbers (simple pattern)
            'phone': re.compile(r'\b(\d{3})[-.]?(\d{3})[-.]?(\d{4})\b'),
            
            # Ranges
            'range': re.compile(r'\b(\d+)\s*[-–—]\s*(\d+)\b'),
            
            # Lists (1,2,3 or 1, 2, 3)
            'list_comma': re.compile(r'\b(\d+)\s*,\s*'),
            
            # Integer with commas (1,000)
            'int_comma': re.compile(r'\b(\d{1,3}(?:,\d{3})+)\b'),
            
            # Plain integers (must be last)
            'integer': re.compile(r'\b(\d+)\b'),
            
            # Units
            'units': re.compile(r'\b(\d+(?:\.\d+)?)\s*(kg|km|m|cm|mm|g|mg|l|ml|sec|min|hr|hrs)\b', re.IGNORECASE),
        }
    
    def normalize(self, text: str, language: str = 'en') -> str:
        """
        Main normalization pipeline
        
        Args:
            text: Raw text from LLM
            language: ISO 639-1 language code
        
        Returns:
            Normalized text ready for TTS
        """
        if not text.strip():
            return text
        
        # Get language for num2words
        lang = self.LANG_MAP.get(language, 'en')
        
        # Apply normalizations in order (specific to general)
        text = self._normalize_percentages(text, lang)
        text = self._normalize_currency(text, lang)
        text = self._normalize_time(text, lang)
        text = self._normalize_dates(text, lang)
        text = self._normalize_ordinals(text, lang)
        text = self._normalize_phone(text, lang)
        text = self._normalize_ranges(text, lang)
        text = self._normalize_units(text, lang)
        text = self._normalize_lists(text, lang)
        text = self._normalize_decimals(text, lang)
        text = self._normalize_integers_with_commas(text, lang)
        text = self._normalize_integers(text, lang)
        text = self._normalize_punctuation(text)
        
        return text
    
    def _num_to_words(self, num: float, lang: str, ordinal: bool = False) -> str:
        """Convert number to words with error handling"""
        try:
            if ordinal:
                return num2words(int(num), lang=lang, to='ordinal')
            return num2words(num, lang=lang)
        except (ValueError, NotImplementedError):
            # Fallback to English if language not supported
            try:
                if ordinal:
                    return num2words(int(num), lang='en', to='ordinal')
                return num2words(num, lang='en')
            except:
                # Last resort: return as string
                return str(num)
    
    def _normalize_percentages(self, text: str, lang: str) -> str:
        """50% → fifty percent"""
        def replace(m):
            num = float(m.group(1))
            words = self._num_to_words(num, lang)
            return f"{words} percent"
        return self.patterns['percentage'].sub(replace, text)
    
    def _normalize_currency(self, text: str, lang: str) -> str:
        """₹100 → one hundred rupees, $50 → fifty dollars"""
        def replace_rs(m):
            num_str = m.group(1).replace(',', '')
            num = float(num_str)
            words = self._num_to_words(num, lang)
            return f"{words} rupees" if lang in ['hi', 'bn', 'gu', 'ta', 'te', 'kn', 'ml'] else f"{words} rupees"
        
        def replace_dollar(m):
            num_str = m.group(1).replace(',', '')
            num = float(num_str)
            words = self._num_to_words(num, lang)
            return f"{words} dollars"
        
        text = self.patterns['currency_rs'].sub(replace_rs, text)
        text = self.patterns['currency_dollar'].sub(replace_dollar, text)
        return text
    
    def _normalize_time(self, text: str, lang: str) -> str:
        """10:30 PM → ten thirty PM, 1530 hrs → fifteen thirty hours"""
        def replace_colon(m):
            hour = int(m.group(1))
            minute = int(m.group(2))
            period = m.group(3) or ''
            
            hour_words = self._num_to_words(hour, lang)
            minute_words = self._num_to_words(minute, lang) if minute != 0 else "o'clock"
            
            if period:
                return f"{hour_words} {minute_words} {period.upper()}"
            return f"{hour_words} {minute_words}"
        
        def replace_hhmm(m):
            hour = int(m.group(1))
            minute = int(m.group(2))
            hour_words = self._num_to_words(hour, lang)
            minute_words = self._num_to_words(minute, lang)
            return f"{hour_words} {minute_words} hours"
        
        text = self.patterns['time_colon'].sub(replace_colon, text)
        text = self.patterns['time_hhmm'].sub(replace_hhmm, text)
        return text
    
    def _normalize_dates(self, text: str, lang: str) -> str:
        """15/01/2026 → fifteenth of january twenty twenty-six"""
        def replace(m):
            day = int(m.group(1))
            month = int(m.group(2))
            year = m.group(3)
            
            day_words = self._num_to_words(day, lang, ordinal=True)
            
            months = ['january', 'february', 'march', 'april', 'may', 'june',
                     'july', 'august', 'september', 'october', 'november', 'december']
            month_name = months[month - 1] if 1 <= month <= 12 else str(month)
            
            year_int = int(year)
            if year_int < 100:
                year_int += 2000 if year_int < 50 else 1900
            
            # Split years like 2026 → twenty twenty-six
            if 1000 <= year_int <= 2999:
                first_two = year_int // 100
                last_two = year_int % 100
                if last_two == 0:
                    year_words = self._num_to_words(first_two, lang) + " hundred"
                else:
                    year_words = f"{self._num_to_words(first_two, lang)} {self._num_to_words(last_two, lang)}"
            else:
                year_words = self._num_to_words(year_int, lang)
            
            return f"{day_words} of {month_name} {year_words}"
        
        text = self.patterns['date_slash'].sub(replace, text)
        text = self.patterns['date_dash'].sub(replace, text)
        return text
    
    def _normalize_ordinals(self, text: str, lang: str) -> str:
        """1st → first, 2nd → second"""
        def replace(m):
            num = int(m.group(1))
            return self._num_to_words(num, lang, ordinal=True)
        return self.patterns['ordinal'].sub(replace, text)
    
    def _normalize_phone(self, text: str, lang: str) -> str:
        """555-123-4567 → five five five one two three four five six seven"""
        def replace(m):
            digits = m.group(1) + m.group(2) + m.group(3)
            words = ' '.join([self._num_to_words(int(d), lang) for d in digits])
            return words
        return self.patterns['phone'].sub(replace, text)
    
    def _normalize_ranges(self, text: str, lang: str) -> str:
        """10-20 → ten to twenty"""
        def replace(m):
            start = int(m.group(1))
            end = int(m.group(2))
            start_words = self._num_to_words(start, lang)
            end_words = self._num_to_words(end, lang)
            return f"{start_words} to {end_words}"
        return self.patterns['range'].sub(replace, text)
    
    def _normalize_units(self, text: str, lang: str) -> str:
        """10 kg → ten kilograms"""
        unit_expansions = {
            'kg': 'kilograms', 'km': 'kilometers', 'm': 'meters',
            'cm': 'centimeters', 'mm': 'millimeters',
            'g': 'grams', 'mg': 'milligrams',
            'l': 'liters', 'ml': 'milliliters',
            'sec': 'seconds', 'min': 'minutes',
            'hr': 'hours', 'hrs': 'hours'
        }
        
        def replace(m):
            num = float(m.group(1))
            unit = m.group(2).lower()
            words = self._num_to_words(num, lang)
            unit_full = unit_expansions.get(unit, unit)
            return f"{words} {unit_full}"
        
        return self.patterns['units'].sub(replace, text)
    
    def _normalize_lists(self, text: str, lang: str) -> str:
        """1,2,3 → one, two, three"""
        def replace(m):
            num = int(m.group(1))
            words = self._num_to_words(num, lang)
            return f"{words}, "
        return self.patterns['list_comma'].sub(replace, text)
    
    def _normalize_decimals(self, text: str, lang: str) -> str:
        """3.5 → three point five"""
        def replace(m):
            integer_part = int(m.group(1))
            decimal_part = m.group(2)
            
            int_words = self._num_to_words(integer_part, lang)
            dec_words = ' '.join([self._num_to_words(int(d), lang) for d in decimal_part])
            
            return f"{int_words} point {dec_words}"
        return self.patterns['decimal'].sub(replace, text)
    
    def _normalize_integers_with_commas(self, text: str, lang: str) -> str:
        """1,000 → one thousand"""
        def replace(m):
            num_str = m.group(1).replace(',', '')
            num = int(num_str)
            return self._num_to_words(num, lang)
        return self.patterns['int_comma'].sub(replace, text)
    
    def _normalize_integers(self, text: str, lang: str) -> str:
        """123 → one hundred twenty-three"""
        def replace(m):
            num = int(m.group(1))
            return self._num_to_words(num, lang)
        return self.patterns['integer'].sub(replace, text)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Clean up extra spaces and normalize punctuation for speech"""
        # Multiple spaces → single space
        text = re.sub(r'\s+', ' ', text)
        
        # Add pauses around punctuation
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        
        # Remove trailing/leading whitespace
        text = text.strip()
        
        return text


# Example usage
if __name__ == "__main__":
    normalizer = TTSTextNormalizer()
    
    test_cases = [
        ("1,2,3 let's begin", "en"),
        ("The temperature is 37.5°C today", "en"),
        ("Meet me at 10:30 PM", "en"),
        ("Call me at 555-123-4567", "en"),
        ("It costs ₹1,500 per month", "hi"),
        ("The date is 15/01/2026", "en"),
        ("I came in 1st place!", "en"),
        ("Run for 10-15 minutes", "en"),
        ("Add 2.5 kg of flour", "en"),
        ("Wait for 30 sec", "en"),
    ]
    
    print("🧪 Text Normalization Test Results:\n")
    for text, lang in test_cases:
        normalized = normalizer.normalize(text, lang)
        print(f"Language: {lang}")
        print(f"Input:    {text}")
        print(f"Output:   {normalized}")
        print("-" * 60)