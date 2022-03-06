
'''
Defines the set of symbols used in text input to the model.
'''
from text import cmudict

custom = True

_pad        = '_'
_punctuation = '!\'(),.:␤? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
if custom:
  _custom ='☺☻♥♦♣♠•◘○◙♂♀♪♫☼►◄↕‼¶§▬↨↑↓→←∟↔▲;'
# ☺     = Anxious
# ☻     = Happy
# ♥     = Angry
# ♦     = Annoyed
# ♣     = Serious
# ♠     = Amused
# ♪     = Singing
# ◘     = Fear
# ○     = Sad
# ◙     = Shouting
# ♂     = Confused
# ♀     = Surprised
# •     = Smug
# ♫     = Love
# ☼     = Sarcastic
# ►     = Tired   OR   Exhausted
# ◄     = Whispering
# ␤     = New Line

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

if custom: # Export all symbols:
  symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet + list(_custom)
else:
  symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

ctc_symbols = [_pad] + list(_letters) + _arpabet
