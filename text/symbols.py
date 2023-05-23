""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
from g2p_en import g2p as grapheme_to_phonem
g2p = grapheme_to_phonem.G2p()

_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
arpa_sym = g2p.phonemes
_arpabet = ["@" + s for s in arpa_sym]



# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + _arpabet

# Special symbol ids
SPACE_ID = symbols.index(" ")
