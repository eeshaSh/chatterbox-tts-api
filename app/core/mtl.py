'''
Multilingual support for the tts model

This is a temporary file to support the multilingual model.
It will be removed when the multilingual model is fully supported.

Currently, the pkuseg package is not compatible, which is required for zh (Chinese). 
So, the SUPPORTED_LANGUAGES constant is overriden from the chatterbox.mtl_tts module to reflect the languages that are supported.
'''

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}