def translate_sentence(model, sentence, src_field, trg_field, device, max_len=50):
    model.eval()
    tokens = [token.text.lower() for token in spacy_de(sentence)]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indices = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    outputs = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        best_guess = output.argmax(2)[:, -1].item()
        outputs.append(best_guess)
        if best_guess == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    return [trg_field.vocab.itos[idx] for idx in outputs]
