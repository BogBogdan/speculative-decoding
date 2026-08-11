import torch

p = torch.tensor([0.5, 0.3, 0.2])
q = torch.tensor([0.2, 0.6, 0.2])


def speculative_decoding():
    print("Speculative decoding is a technique used in natural language processing (NLP) to generate text by predicting the next word or token based on the context of the previous words. It involves using a language model to generate multiple possible continuations of a given input, and then selecting the most likely continuation based on a scoring function. This approach can help improve the quality and diversity of generated text, as it allows for exploration of different possibilities rather than relying solely on a single deterministic output.")

def tokenize_target(prefixes):
    print("Tokenization is the process of breaking down text into smaller units called tokens, which can be words, subwords, or characters. In NLP, tokenization is an essential step for preparing text data for analysis and modeling. A small tokenizer typically refers to a tokenizer that uses a limited vocabulary or a simplified tokenization scheme, which can be useful for specific applications or resource-constrained environments. Small tokenizers may sacrifice some accuracy or expressiveness in favor of efficiency and speed.")

def tokenize_draft(prefixes):
    print("Tokenization is the process of breaking down text into smaller units called tokens, which can be words, subwords, or characters. In NLP, tokenization is an essential step for preparing text data for analysis and modeling. A draft tokenizer typically refers to a tokenizer that is in the early stages of development or experimentation, and may not yet have been fully optimized or validated. Draft tokenizers may be used for testing new tokenization strategies or for exploring different approaches to text processing.")

def speculative_sampling(prefixes):
    tokens_draft = tokenize_draft(prefixes)
    tokens_target = tokenize_target(prefixes)

    if len(tokens_draft) != len(tokens_target):
        raise ValueError("Tokenized draft and target must have the same length.")

    results = []
    for draft_token, target_token in zip(tokens_draft, tokens_target):
        results.append(max(0, target_token-draft_token))

    
     
    