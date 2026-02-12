from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast
import os

# 🔤 Vocabulari de caràcters catalans complet
vocab_list = list(
    "abcdefghijklmnopqrstuvwxyzçàèéíïòóúü" +
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÇÀÈÉÍÏÒÓÚÜ" +
    " ,.;·'-?!\""
)
vocab_list += ["<ctc_blank>", "<pad>", "<unk>"]

# Diccionari {caràcter: ID}
vocab_dict = {c: i for i, c in enumerate(vocab_list)}

# Tokenizer WordLevel per caràcter
tokenizer = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="<unk>"))
tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

# HuggingFace wrapper
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    pad_token="<pad>"
)

# Desa
save_dir = "tokenizers/ctc_catalan_char_tokenizer"
os.makedirs(save_dir, exist_ok=True)
hf_tokenizer.save_pretrained(save_dir)

# Test
text = "L'ús d'algoritmes al sector públic."
encoded = hf_tokenizer(text)
decoded = ''.join(hf_tokenizer.convert_ids_to_tokens(encoded.input_ids))

print("Token IDs:", encoded.input_ids)
print("Decoded:", decoded)

# Mostra l'ID real del blank
blank_id = hf_tokenizer.convert_tokens_to_ids("<ctc_blank>")
print("CTC blank ID:", blank_id)
