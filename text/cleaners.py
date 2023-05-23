""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from phonemizer import phonemize
import phonemizer
from .arpa import ARPAPhonemizer

global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
spanish_phonemizer = phonemizer.backend.EspeakBackend(language='es-419', preserve_punctuation=True,  with_stress=True)

fest_phonemizer = phonemizer.backend.FestivalBackend(language='en-us', preserve_punctuation=True)
arpa_phonemizer = ARPAPhonemizer()




# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes


def english_cleaners2(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)
  phonemes = phonemes[0]
  phonemes = collapse_whitespace(phonemes)
  return phonemes

def f_english_cleaners2(text):
  '''Pipeline for English text with Festival phonemizer, including abbreviation expansion. + punctuation'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = fest_phonemizer.phonemize([text], strip=True, njobs=1)
  phonemes = phonemes[0]
  phonemes = collapse_whitespace(phonemes)
  return phonemes

def arpa_precleaners(text):
  '''Pipeline for English text before ARPA'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)

  return text

def arpa_cleaners(text):
  '''Pipeline for ARPA English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = arpa_phonemizer.doARPA(text)

  return phonemes

def spanish_cleaners(text):
  '''Pipeline for Spanish text, punctuation + stress'''
  text = lowercase(text)
  phonemes = spanish_phonemizer.phonemize([text], strip=True, njobs=1)
  phonemes = phonemes[0].replace("(es-419)","").replace("(en)","")
  phonemes = collapse_whitespace(phonemes)
  return phonemes