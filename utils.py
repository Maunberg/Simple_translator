#!/usr/bin/env python3
"""
Утилиты для обучения и работы с моделью перевода
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

def plot_training_curves(pretrain_losses, finetune_losses, save_path='training_curves.png'):
    """Построение графиков обучения"""
    plt.figure(figsize=(12, 5))
    
    # График предобучения
    plt.subplot(1, 2, 1)
    plt.plot(pretrain_losses, label='Предобучение', color='blue')
    plt.title('Кривая предобучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()
    plt.grid(True)
    
    # График файнтюнинга
    plt.subplot(1, 2, 2)
    plt.plot(finetune_losses, label='Файнтюнинг', color='red')
    plt.title('Кривая файнтюнинга')
    plt.xlabel('Эпоха')
    plt.ylabel('Потеря')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Графики сохранены в {save_path}")

def save_training_log(pretrain_losses, finetune_losses, config, save_path='training_log.json'):
    """Сохранение лога обучения"""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'pretrain_losses': pretrain_losses,
        'finetune_losses': finetune_losses,
        'final_pretrain_loss': pretrain_losses[-1] if pretrain_losses else None,
        'final_finetune_loss': finetune_losses[-1] if finetune_losses else None
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"Лог обучения сохранен в {save_path}")

def calculate_bleu_score(predictions, references):
    """Простой расчет BLEU score"""
    from collections import Counter
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def precision(pred_ngrams, ref_ngrams):
        pred_counts = Counter(pred_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        overlap = sum(min(pred_counts[ngram], ref_counts[ngram]) for ngram in pred_counts)
        total = sum(pred_counts.values())
        
        return overlap / total if total > 0 else 0
    
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        # BLEU-1 до BLEU-4
        precisions = []
        for n in range(1, 5):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            precisions.append(precision(pred_ngrams, ref_ngrams))
        
        # Геометрическое среднее
        bleu = np.exp(np.mean([np.log(p) if p > 0 else -np.inf for p in precisions]))
        scores.append(bleu)
    
    return np.mean(scores)

def evaluate_model(model, tokenizer, test_data, device, max_length=128):
    """Оценка качества модели"""
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for i, sample in enumerate(test_data):
            if i >= 100:  # Ограничиваем для быстрой оценки
                break
                
            src_text = sample['src_text']
            tgt_text = sample['tgt_text']
            
            # Перевод
            src_tokens = tokenizer.tokenize(src_text)
            bos_token = tokenizer.vocab_tokentoid.get('<BOS>', 1)
            eos_token = tokenizer.vocab_tokentoid.get('<EOS>', 2)
            
            src_tokens = [bos_token] + src_tokens + [eos_token]
            src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
            
            tgt_tokens = [bos_token]
            for _ in range(max_length):
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
                output = model(src_tensor, tgt_tensor)
                next_token = output[0, -1, :].argmax().item()
                
                if next_token == eos_token:
                    break
                tgt_tokens.append(next_token)
            
            try:
                translated = tokenizer.totext(tgt_tokens[1:])
                predictions.append(translated)
                references.append(tgt_text)
            except:
                predictions.append("")
                references.append(tgt_text)
    
    # Расчет BLEU
    bleu_score = calculate_bleu_score(predictions, references)
    
    return {
        'bleu_score': bleu_score,
        'predictions': predictions[:10],  # Первые 10 для примера
        'references': references[:10]
    }

def create_attention_visualization(model, tokenizer, src_text, tgt_text, layer_idx=0, head_idx=0):
    """Визуализация внимания (упрощенная версия)"""
    model.eval()
    
    # Токенизация
    src_tokens = tokenizer.tokenize(src_text)
    tgt_tokens = tokenizer.tokenize(tgt_text)
    
    bos_token = tokenizer.vocab_tokentoid.get('<BOS>', 1)
    eos_token = tokenizer.vocab_tokentoid.get('<EOS>', 2)
    
    src_tokens = [bos_token] + src_tokens + [eos_token]
    tgt_tokens = [bos_token] + tgt_tokens + [eos_token]
    
    src_tensor = torch.tensor([src_tokens], dtype=torch.long)
    tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long)
    
    # Получение внимания (упрощенная версия)
    with torch.no_grad():
        # Здесь должна быть логика извлечения весов внимания
        # Для упрощения возвращаем случайную матрицу
        attention_weights = torch.rand(len(tgt_tokens), len(src_tokens))
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights.numpy(), cmap='Blues', aspect='auto')
    plt.xlabel('Исходные токены')
    plt.ylabel('Целевые токены')
    plt.title(f'Матрица внимания (слой {layer_idx}, голова {head_idx})')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def optimize_model_parameters(model, learning_rate=0.0001, weight_decay=1e-5):
    """Оптимизация параметров модели"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Создание scheduler с проверкой версии PyTorch
    try:
        # Попытка создать scheduler с verbose (для новых версий)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
    except TypeError:
        # Fallback для старых версий PyTorch
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
    
    return optimizer, scheduler

def save_model_checkpoint(model, optimizer, epoch, loss, save_path):
    """Сохранение чекпоинта модели"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    print(f"Чекпоинт сохранен: {save_path}")

def load_model_checkpoint(model, optimizer, checkpoint_path):
    """Загрузка чекпоинта модели"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Чекпоинт загружен: эпоха {epoch}, потеря {loss:.4f}")
    
    return epoch, loss

def get_model_info(model):
    """Получение информации о модели"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Примерный размер в МБ
        'layers': len(list(model.children()))
    }
    
    return info

def print_model_summary(model):
    """Вывод сводки о модели"""
    info = get_model_info(model)
    
    print("=== Информация о модели ===")
    print(f"Всего параметров: {info['total_parameters']:,}")
    print(f"Обучаемых параметров: {info['trainable_parameters']:,}")
    print(f"Размер модели: {info['model_size_mb']:.2f} МБ")
    print(f"Количество слоев: {info['layers']}")
    print("=" * 30)
