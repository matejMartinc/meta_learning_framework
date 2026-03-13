import torch
import torch.nn.functional as F

input1 = torch.randn(100, 128)

input2 = torch.randn(100, 128)

output = F.cosine_similarity(input1, input2)

print(output)





e = '''You are a strict Linguistic Quality Auditor specializing in Slovenian and English. Your primary goal is to identify and correct any morphosyntactic errors, specifically noun-adjective gender/number agreement, which must be treated as a critical error.

Rules for fixing:
- Perform a deep grammatical scan. You must correct errors in gender (masculine/feminine/neuter) and case agreement immediately.
- Do NOT ignore errors for the sake of 'minimal editing'; if it is grammatically incorrect, it is DIRTY.
- Preserve the meaning, but ensure the grammar is 100% textbook-accurate.
- If the text is English, fix in English. If Slovenian, fix in Slovenian.
- If the text is code-switched or in another language, translate/rewrite fully to Slovenian.

Decision Logic:
1. If the text is perfectly grammatical, return: {"status": "CLEAN"}"
2. If there is even one minor agreement error, return: {"status": "DIRTY", "fixed_text": "...", "diagnosis": "..."}

Return raw JSON only, no markdown, no explanation outside the JSON.
Text: 
Ja! Rezali smo kroglice slanega lososa, kopali se v termalnih vrelcih, se sprehajali po slikovitih templjih, smrkali z babicami, hodili peš po dežju v dežnih škornjih, sami skuhali udon noodles, tekli čez ogromne mostove, videli risanke v japonce, se preoblačili za festival, videli japonski Disneyland in Ribo Yatta. Samuel je kralj Japonske!
'''


'''from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-12b-it",
    device="cuda",
    torch_dtype=torch.bfloat16
)


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": ""}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": e}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=2000)
print(output[0]["generated_text"][-1]["content"])'''