import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_text(input_text, model, tokenizer, n=5):
    """
    Generates text iteratively by feeding the output back into the model.

    :param input_text: Initial text to start the generation.
    :param model: Pre-trained GPT-2 model.
    :param tokenizer: Pre-trained GPT-2 tokenizer.
    :param n: Number of iterations for text generation.
    :return: Generated text.
    """
    model.eval()
    generated_text = input_text

    for _ in range(n):
        indexed_tokens = tokenizer.encode(generated_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        if torch.cuda.is_available():
            tokens_tensor = tokens_tensor.to('cuda')

        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        generated_text = tokenizer.decode(indexed_tokens + [predicted_index])

    return generated_text

# Check if CUDA is available and print device info
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("torch run on cpu")

# Load pre-trained model tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# If you have a GPU, put everything on cuda
if torch.cuda.is_available():
    model.to('cuda')


while True:
    # Initial text input
    # text = "What is the fastest car in the country of"
    text = input()

    # Generate text iteratively
    generated_text = generate_text(text, model, tokenizer, n=15)

    # Print the generated text
    print(generated_text)
