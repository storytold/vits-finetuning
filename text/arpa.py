from g2p_en import G2p
import gdown
import os

class ARPAPhonemizer():
    def __init__(self, dict_override = ""):
        dict_fn = "zdict.txt" if not len(dict_override) else dict_override
        
        if not len(dict_override) and not os.path.isfile(dict_fn):
            print("Downloading ARPA dict..")
            gdown.download("https://drive.google.com/file/d/1lYKcml_bkTiOke2TMF3F_fn7UWTYHNWp/view?usp=share_link", dict_fn, fuzzy=True, quiet=False)
        
        self.arpa_dict = self._load_dict(dict_fn)
        self.g2p = G2p()

    
    def _load_dict(self, in_dpath):
      dict_ret = {}

      with open(in_dpath,encoding="utf-8-sig") as f:
        d_lines = f.readlines()

      last_word = ""
      for d_li in d_lines:
        splits = d_li.strip().split("\t")

        word = splits[0]
        pron = splits[1]

        #only accept 1st entry for each word
        if word == last_word:
          continue

        dict_ret[word] = pron
        last_word = word

      return dict_ret

    def _g2pfil(self, intxt):
      sta = self.g2p(intxt)
      ret = " ".join(sta)
      return ret
    
    def doARPA(self, text):
        if text[0] == '{':
          return text
        out = ''
        for word_ in text.split(" "):
            word=word_; end_chars = ''
            while any(elem in word for elem in r"!?,.;") and len(word) > 1:
                if word[-1] == '!': end_chars = '!' + end_chars; word = word[:-1]
                if word[-1] == '?': end_chars = '?' + end_chars; word = word[:-1]
                if word[-1] == ',': end_chars = ',' + end_chars; word = word[:-1]
                if word[-1] == '.': end_chars = '.' + end_chars; word = word[:-1]
                if word[-1] == ';': end_chars = ';' + end_chars; word = word[:-1]
                else: break
            try: word_arpa = self.arpa_dict[word.lower()]
            except: word_arpa = self._g2pfil(word)
            if len(word_arpa)!=0: word = "{" + str(word_arpa) + "}"
            out = (out + " " + word + end_chars).strip()

        return out

