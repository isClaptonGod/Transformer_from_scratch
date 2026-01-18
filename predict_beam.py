import torch
import torch.nn.functional as F

def beam_search(model, src_sentence, src_field, trg_field, device, beam_size=3, max_len=50):
    model.eval()
    
    # 1. Preprocess source sentence
    tokens = [token.text.lower() for token in spacy_de(src_sentence)]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indices = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    # 2. Get Encoder output (only once)
    with torch.no_grad():
        src_mask = model.make_src_mask(src_tensor)
        enc_src = model.dropout(model.pos_encoding(model.encoder_embedding(src_tensor)))
        for layer in model.encoder_layers:
            enc_src = layer(enc_src, enc_src, enc_src, src_mask)

    # 3. Initialize beam: (score, sequence)
    # Start with <sos>
    beams = [(0.0, [trg_field.vocab.stoi[trg_field.init_token]])]
    completed_sentences = []

    for _ in range(max_len):
        new_beams = []
        for score, seq in beams:
            # If sequence ends in <eos>, move to completed
            if seq[-1] == trg_field.vocab.stoi[trg_field.eos_token]:
                completed_sentences.append((score, seq))
                continue

            # Predict next token
            trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(src_tensor, trg_tensor) # You might need to adjust forward to accept enc_src directly for efficiency
                # Log probabilities for the last token
                log_probs = F.log_softmax(output[0, -1, :], dim=-1)
            
            # Get top k candidates
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                new_beams.append((score + topk_log_probs[i].item(), seq + [topk_indices[i].item()]))

        # Sort and keep top k
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        
        # If all beams are completed, stop
        if not beams: break

    # Combine and pick the best one
    completed_sentences.extend(beams)
    best_score, best_seq = max(completed_sentences, key=lambda x: x[0] / len(x[1])) # Length normalized score
    
    return [trg_field.vocab.itos[idx] for idx in best_seq]
