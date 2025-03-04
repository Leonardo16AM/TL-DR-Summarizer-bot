from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import math

#region calculate_perplexity
def calculate_perplexity(model_name, text):
    # Carga el modelo y el tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.eval()

  
    inputs = tokenizer(text, return_tensors="pt")
    response_tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    response_input_ids = response_tokens["input_ids"]
    with torch.no_grad():
        response_outputs = model(input_ids=response_input_ids, labels=response_input_ids)
        response_loss = response_outputs.loss

    perplexity = math.exp(response_loss.item())
    return perplexity

if __name__ == "__main__":
    model_name = "datificate/gpt2-small-spanish"  
    mensaje_B = "¿Tienes algun problema mental?"
    mensaje_A = "Si."

    perplexity = calculate_perplexity(model_name, mensaje_B, mensaje_A)
    print(f"Perplexity de la respuesta: {perplexity}")

    if perplexity < 50:
        print("La respuesta parece coherente.")
    else:
        print("La respuesta no parece coherente.")
