"""
Preprocessing utilities for Nepali hate speech detection
"""
import re
import emoji
import regex

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False
    print("Warning: indic_transliteration not available. Roman/Devanagari conversion disabled.")

# Nepali stopwords
NEPALI_STOPWORDS = set([
    "र", "मा", "कि", "भने", "त", "छ", "हो", "लाई", "ले",
    "गरेको", "गर्छ", "गर्छन्", "हुन्", "गरे", "न", "नभएको",
    "को", "का", "की", "ने", "पनि", "नै", "थियो", "थिए"
])

# Dirghikaran normalization mapping
DIRGHIKARAN_MAP = {
    "उ": "ऊ", "इ": "ई", "ऋ": "रि", "ए": "ऐ", "अ": "आ",
    "\u200d": "", "\u200c": "",  # Zero-width characters
    "।": ".", "॥": ".",  # Devanagari punctuation
    "ि": "ी", "ु": "ू"  # Vowel signs
}

# Cache for Roman stopwords
_roman_stopwords_cache = None


def is_devanagari(text: str) -> bool:
    """
    Detect if text contains significant Devanagari characters.
    Returns True if >50% of letters are Devanagari.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    
    devanagari_chars = len(regex.findall(r'\p{Devanagari}', text))
    total_chars = len(regex.findall(r'\p{L}', text))  # All letters
    
    if total_chars == 0:
        return False
    return (devanagari_chars / total_chars) > 0.5


def devanagari_to_roman(text: str) -> str:
    """Convert Devanagari script to Roman (ITRANS)."""
    if not TRANSLITERATION_AVAILABLE:
        return text
    try:
        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception:
        return text


def roman_to_devanagari(text: str) -> str:
    """
    Convert Roman script to Devanagari (ITRANS).
    Note: This is kept for ML/GRU compatibility but NOT recommended for transformers.
    """
    if not TRANSLITERATION_AVAILABLE:
        return text
    try:
        return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    except Exception:
        return text


def normalize_dirghikaran(text: str) -> str:
    """Apply dirghikaran normalization to reduce orthographic variants."""
    for original, replacement in DIRGHIKARAN_MAP.items():
        text = text.replace(original, replacement)
    return text


def clean_text(text: str, aggressive: bool = True) -> str:
    """
    Clean text with various preprocessing steps.
    
    Args:
        text: Input text
        aggressive: If True, removes punctuation and numbers
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace="")
    
    if aggressive:
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        
        # Remove punctuation (keep Devanagari characters)
        text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def remove_stopwords_devanagari(text: str) -> str:
    """Remove Devanagari stopwords."""
    words = text.split()
    filtered = [w for w in words if w not in NEPALI_STOPWORDS]
    return ' '.join(filtered)


def remove_stopwords_roman(text: str) -> str:
    """Remove Romanized stopwords."""
    global _roman_stopwords_cache
    
    # Initialize cache on first use
    if _roman_stopwords_cache is None:
        _roman_stopwords_cache = set([
            devanagari_to_roman(w) for w in NEPALI_STOPWORDS
        ])
    
    words = text.split()
    filtered = [w for w in words if w not in _roman_stopwords_cache]
    return ' '.join(filtered)


def preprocess_for_ml_gru(text: str) -> str:
    """
    Preprocess for ML/GRU models: Romanized, cleaned, stopwords removed.
    
    Pipeline:
    1. Clean text
    2. Remove stopwords (script-dependent)
    3. Normalize dirghikaran (if Devanagari)
    4. Transliterate to Roman
    """
    if not isinstance(text, str):
        return ""
    
    if is_devanagari(text):
        text = clean_text(text, aggressive=True)
        text = normalize_dirghikaran(text)
        text = remove_stopwords_devanagari(text)
        text = devanagari_to_roman(text)
    else:
        text = clean_text(text, aggressive=True)
        text = remove_stopwords_roman(text)
    
    return text


def preprocess_for_transformer(text: str, keep_mixed_scripts: bool = True) -> str:
    """
    Preprocess for Transformer models (XLM-RoBERTa): Keep both scripts.
    
    RECOMMENDED: Use keep_mixed_scripts=True (default)
    - XLM-RoBERTa handles both Devanagari and Roman scripts natively
    - Avoids transliteration errors
    - Preserves natural code-switching behavior
    - Better performance in practice
    
    Pipeline (keep_mixed_scripts=True):
    1. Light cleaning (preserve structure for tokenizer)
    2. Normalize only Devanagari portions
    
    Pipeline (keep_mixed_scripts=False - legacy):
    1. Transliterate Roman → Devanagari (not recommended)
    2. Light cleaning
    3. Normalize dirghikaran
    
    Args:
        text: Input text
        keep_mixed_scripts: If True, keep both scripts (RECOMMENDED for XLM-RoBERTa)
                          If False, force to Devanagari (legacy, not recommended)
    
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    if keep_mixed_scripts:
        # RECOMMENDED APPROACH: Keep both scripts
        # Light cleaning (preserve punctuation for tokenizer)
        text = clean_text(text, aggressive=False)
        
        # Only normalize Devanagari portions
        # Process word by word to avoid normalizing Roman text
        words = text.split()
        processed_words = []
        for word in words:
            if is_devanagari(word):
                word = normalize_dirghikaran(word)
            processed_words.append(word)
        text = ' '.join(processed_words)
    else:
        # LEGACY APPROACH: Force to Devanagari (not recommended)
        # Convert to Devanagari if needed
        if not is_devanagari(text) and TRANSLITERATION_AVAILABLE:
            text = roman_to_devanagari(text)
        
        # Light cleaning (preserve punctuation)
        text = clean_text(text, aggressive=False)
        
        # Normalize orthographic variants
        text = normalize_dirghikaran(text)
    
    return text


def batch_preprocess(texts, mode='transformer', **kwargs):
    """
    Batch preprocess texts.
    
    Args:
        texts: List of text strings
        mode: 'ml' for ML/GRU, 'transformer' for XLM-RoBERTa
        **kwargs: Additional arguments passed to preprocessing functions
                 For transformer mode: keep_mixed_scripts (default: True)
        
    Returns:
        List of preprocessed texts
    """
    if mode == 'ml':
        return [preprocess_for_ml_gru(t) for t in texts]
    elif mode == 'transformer':
        keep_mixed = kwargs.get('keep_mixed_scripts', True)
        return [preprocess_for_transformer(t, keep_mixed_scripts=keep_mixed) for t in texts]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'ml' or 'transformer'")


# ============================================================================
# USAGE EXAMPLES AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("NEPALI PREPROCESSING - TESTING")
    print("="*70)
    
    test_texts = [
        "Musalman haru aatankbadi hun",
        "यो नेपाली पाठ हो",
        "Nepal ko राजधानी Kathmandu ho",
        "Hindu ra Muslim हरू sabaile samman garnu parcha",
        "@user123 check this https://example.com 🔥",
    ]
    
    print("\n1. TRANSFORMER PREPROCESSING (RECOMMENDED - Mixed Scripts)")
    print("-"*70)
    for text in test_texts:
        processed = preprocess_for_transformer(text, keep_mixed_scripts=True)
        print(f"Original:    {text}")
        print(f"Processed:   {processed}")
        print()
    
    print("\n2. TRANSFORMER PREPROCESSING (Legacy - Force Devanagari)")
    print("-"*70)
    for text in test_texts[:2]:  # Just show 2 examples
        processed = preprocess_for_transformer(text, keep_mixed_scripts=False)
        print(f"Original:    {text}")
        print(f"Processed:   {processed}")
        print()
    
    print("\n3. ML/GRU PREPROCESSING (Romanized)")
    print("-"*70)
    for text in test_texts[:2]:  # Just show 2 examples
        processed = preprocess_for_ml_gru(text)
        print(f"Original:    {text}")
        print(f"Processed:   {processed}")
        print()
    
    print("\n" + "="*70)
    print("RECOMMENDATION FOR XLM-RoBERTa:")
    print("Use: preprocess_for_transformer(text, keep_mixed_scripts=True)")
    print("="*70)