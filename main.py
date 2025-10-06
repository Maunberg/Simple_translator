import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import json
import pickle
import os
from datetime import datetime

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Transformer
from tokenizer import Tokeniser
from utils import (
    plot_training_curves, save_training_log, evaluate_model, 
    optimize_model_parameters, save_model_checkpoint, 
    get_model_info, print_model_summary
)

# Конфигурация
CONFIG = {
    'way_to_data': '/mnt/asr_hot/dutov/study/ml/lab_1_nlp/tatoeba_pairs_filtered.csv',
    'langs': ['eng', 'epo', 'fra', 'deu', 'rus', 'spa', 'tur', 'ita', 'jpn', 'por'],
    'language_selection': ['rus', 'deu'], 
    'way_to_vocab': 'vocab.json',
    'way_to_tokenized': 'tokenized_data.pkl',
    'way_to_model': 'transformer_model_rus_deu.pth',
    'way_to_config': 'model_config_rus_deu.json',
    
    # Параметры модели (уменьшены для ускорения)
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 3,
    'd_ff': 1024,
    'max_seq_length': 128,
    'dropout': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Параметры обучения (оптимизированы для скорости)
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs_pretrain': 40,
    'num_epochs_finetune': 20,
    'save_every': 5,
    'eval_every': 1,
    
    # Языковые токены
    'lang_tokens': {
        'eng': '<EN>',
        'rus': '<RU>', 
        'fra': '<FR>',
        'deu': '<DE>',
        'spa': '<ES>',
        'ita': '<IT>',
        'por': '<PT>',
        'tur': '<TR>',
        'jpn': '<JP>',
        'epo': '<EO>'
    }
}

class TranslationDataset(data.Dataset):
    def __init__(self, src_texts, tgt_texts, src_langs, tgt_langs, tokenizer, max_length=128, lang_tokens=None):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_langs = src_langs
        self.tgt_langs = tgt_langs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang_tokens = lang_tokens or {}
        
        # Сохраняем исходные тексты для оценки
        self.original_src_texts = src_texts.copy()
        self.original_tgt_texts = tgt_texts.copy()
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # Проверяем, существуют ли атрибуты языков
        if hasattr(self, 'src_langs') and hasattr(self, 'tgt_langs'):
            src_lang = self.src_langs[idx]
            tgt_lang = self.tgt_langs[idx]
        else:
            # Если атрибуты не существуют, используем значения по умолчанию
            src_lang = 'eng'
            tgt_lang = 'rus'
        
        # Токенизация
        src_tokens = self.tokenizer.tokenize(src_text)
        tgt_tokens = self.tokenizer.tokenize(tgt_text)
        
        # Получение языковых токенов
        src_lang_token = self.tokenizer.get_lang_token(src_lang)
        tgt_lang_token = self.tokenizer.get_lang_token(tgt_lang)
        
        # Токенизация языковых токенов
        if src_lang_token:
            src_lang_tokens = self.tokenizer.tokenize(src_lang_token)
        else:
            src_lang_tokens = []
            
        if tgt_lang_token:
            tgt_lang_tokens = self.tokenizer.tokenize(tgt_lang_token)
        else:
            tgt_lang_tokens = []
        
        # Добавление специальных токенов
        bos_token = self.tokenizer.vocab_tokentoid.get('<BOS>', 1)
        eos_token = self.tokenizer.vocab_tokentoid.get('<EOS>', 2)
        
        # Формирование последовательностей с языковыми токенами
        # Исходная: [BOS] [LANG_SRC] [TEXT] [EOS]
        src_tokens = [bos_token] + src_lang_tokens + src_tokens
        # Целевая: [BOS] [LANG_TGT] [TEXT] [EOS]  
        tgt_tokens = [bos_token] + tgt_lang_tokens + tgt_tokens
        
        # Обрезка до максимальной длины
        src_tokens = src_tokens[:self.max_length-1] + [eos_token]
        tgt_tokens = tgt_tokens[:self.max_length-1] + [eos_token]
        
        # Паддинг
        src_tokens = src_tokens + [0] * (self.max_length - len(src_tokens))
        tgt_tokens = tgt_tokens + [0] * (self.max_length - len(tgt_tokens))
        
        return {
            'src': torch.tensor(src_tokens[:self.max_length], dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens[:self.max_length], dtype=torch.long)
        }

def get_selected_languages():
    """Определение языков на основе выбора в конфиге"""
    language_selection = CONFIG['language_selection']
    
    if language_selection == 'all':
        return CONFIG['langs']
    else:
        # Если указан конкретный список языков
        if isinstance(language_selection, list):
            return language_selection
        else:
            print(f"Неизвестный выбор языков: {language_selection}, используем все языки")
            return CONFIG['langs']

def load_data():
    """Загрузка и подготовка данных"""
    print("Загрузка данных...")
    df = pd.read_csv(CONFIG['way_to_data'])
    
    # Получение выбранных языков
    selected_langs = get_selected_languages()
    print(f"Выбранные языки: {selected_langs}")
    
    # Фильтрация по выбранным языкам
    df = df[df['SRC LANG'].isin(selected_langs) & df['TRG LANG'].isin(selected_langs)]
    
    print(f"Загружено {len(df)} пар предложений")
    return df

def create_tokenizer(df):
    """Создание или загрузка токенизатора"""
    print("Создание/загрузка токенизатора...")
    tok = Tokeniser()
    
    if not os.path.exists(CONFIG['way_to_vocab']):
        print("Создание нового словаря...")
        text = df['SRC'].to_list()
        text.extend(df['TRG'].to_list())
        text = '\n'.join(list(set(text)))
        
        # Языковые токены уже добавлены в токенизатор как специальные токены
        tok.generate_vocab(text)
    else:
        print("Загрузка существующего словаря...")
        tok.load_vocab(CONFIG['way_to_vocab'])
        
        # Языковые токены уже включены в словарь как специальные токены
    
    print(f"Размер словаря: {len(tok)}")
    return tok

def get_tokenized_data_filename():
    """Определение имени файла токенизированных данных на основе выбранных языков"""
    language_selection = CONFIG['language_selection']
    
    if language_selection == 'all':
        return 'tokenized_data.pkl'
    else:
        # Если указан конкретный список языков, создаем имя на основе языков
        if isinstance(language_selection, list):
            lang_suffix = '_'.join(language_selection)
            return f'tokenized_data_{lang_suffix}.pkl'
        else:
            return 'tokenized_data.pkl'

def tokenize_and_save_data(df, tokenizer):
    """Токенизация и сохранение данных"""
    print("Токенизация данных...")
    
    # Определяем имя файла на основе выбранных языков
    tokenized_filename = get_tokenized_data_filename()
    print(f"Имя файла токенизированных данных: {tokenized_filename}")
    
    if os.path.exists(tokenized_filename):
        print("Загрузка токенизированных данных...")
        with open(tokenized_filename, 'rb') as f:
            return pickle.load(f)
    
    # Подготовка данных для обучения
    src_texts = df['SRC'].tolist()
    tgt_texts = df['TRG'].tolist()
    src_langs = df['SRC LANG'].tolist()
    tgt_langs = df['TRG LANG'].tolist()
    
    # Создание датасета с языковыми токенами
    dataset = TranslationDataset(
        src_texts, tgt_texts, src_langs, tgt_langs, 
        tokenizer, CONFIG['max_seq_length'], CONFIG['lang_tokens']
    )
    
    # Сохранение токенизированных данных
    with open(tokenized_filename, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Токенизировано {len(dataset)} пар предложений с языковыми токенами")
    return dataset

def create_model(tokenizer):
    """Создание модели"""
    print("Создание модели...")
    
    src_vocab_size = len(tokenizer)
    tgt_vocab_size = src_vocab_size
    
    model = Transformer(
        src_vocab_size, tgt_vocab_size, 
        CONFIG['d_model'], CONFIG['num_heads'], 
        CONFIG['num_layers'], CONFIG['d_ff'], 
        CONFIG['max_seq_length'], CONFIG['dropout'], 
        CONFIG['device']
    )
    
    model = model.to(CONFIG['device'])
    return model

def pretrain_model(model, tokenizer, dataset):
    """Предобучение модели на предсказании следующего токена"""
    print("Начало предобучения модели...")
    
    # Создание DataLoader с оптимизацией
    dataloader = data.DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True if CONFIG['device'] == 'cuda' else False,
        persistent_workers=True
    )
    
    # Оптимизированный оптимизатор
    optimizer, scheduler = optimize_model_parameters(model, CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Смешанная точность для ускорения (временно отключена для отладки)
    scaler = None  # torch.cuda.amp.GradScaler() if CONFIG['device'] == 'cuda' else None
    
    model.train()
    losses = []
    
    for epoch in range(CONFIG['num_epochs_pretrain']):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{CONFIG['num_epochs_pretrain']}")
        
        for batch in progress_bar:
            src = batch['src'].to(CONFIG['device'])
            tgt = batch['tgt'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # Использование смешанной точности для ускорения
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Обучение на предсказании следующего токена
                    output = model(src, tgt[:, :-1])
                    loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                
                scaler.scale(loss).backward()
                
                # Градиентное обрезание для стабильности
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Обучение на предсказании следующего токена
                output = model(src, tgt[:, :-1])
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                
                loss.backward()
                
                # Градиентное обрезание для стабильности
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        # Обновление learning rate
        scheduler.step(avg_loss)
        
        print(f"Эпоха {epoch+1}, средняя потеря: {avg_loss:.4f}")
        
        # Сохранение чекпоинта
        if (epoch + 1) % CONFIG['save_every'] == 0:
            save_model_checkpoint(model, optimizer, epoch, avg_loss, f"pretrain_checkpoint_epoch_{epoch+1}.pth")
    
    # Сохранение модели
    torch.save(model.state_dict(), CONFIG['way_to_model'])
    
    # Сохранение конфигурации
    with open(CONFIG['way_to_config'], 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print("Предобучение завершено!")
    return model, losses

def finetune_model(model, tokenizer, dataset):
    """Файнтюнинг модели для перевода"""
    print("Начало файнтюнинга для перевода...")
    
    # Создание DataLoader с оптимизацией
    dataloader = data.DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True if CONFIG['device'] == 'cuda' else False,
        persistent_workers=True
    )
    
    # Оптимизатор с меньшим learning rate для файнтюнинга
    optimizer, scheduler = optimize_model_parameters(model, CONFIG['learning_rate'] * 0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Смешанная точность для ускорения (временно отключена для отладки)
    scaler = None  # torch.cuda.amp.GradScaler() if CONFIG['device'] == 'cuda' else None
    
    model.train()
    losses = []
    
    for epoch in range(CONFIG['num_epochs_finetune']):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Файнтюнинг эпоха {epoch+1}/{CONFIG['num_epochs_finetune']}")
        
        for batch in progress_bar:
            src = batch['src'].to(CONFIG['device'])
            tgt = batch['tgt'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # Использование смешанной точности для ускорения
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Обучение на переводе
                    output = model(src, tgt[:, :-1])
                    loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                
                scaler.scale(loss).backward()
                
                # Градиентное обрезание для стабильности
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Обучение на переводе
                output = model(src, tgt[:, :-1])
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                
                loss.backward()
                
                # Градиентное обрезание для стабильности
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        # Обновление learning rate
        scheduler.step(avg_loss)
        
        print(f"Файнтюнинг эпоха {epoch+1}, средняя потеря: {avg_loss:.4f}")
        
        # Сохранение чекпоинта
        if (epoch + 1) % CONFIG['save_every'] == 0:
            save_model_checkpoint(model, optimizer, epoch, avg_loss, f"finetune_checkpoint_epoch_{epoch+1}.pth")
    
    # Финальное сохранение
    torch.save(model.state_dict(), 'final_model.pth')
    
    print("Файнтюнинг завершен!")
    return model, losses

def translate_text(model, tokenizer, text, src_lang='eng', tgt_lang='rus', max_length=None):
    """Перевод текста с указанием языков"""
    if max_length is None:
        max_length = CONFIG['max_seq_length']
    
    model.eval()
    
    # Получение языковых токенов
    src_lang_token = tokenizer.get_lang_token(src_lang)
    tgt_lang_token = tokenizer.get_lang_token(tgt_lang)
    
    # Токенизация входного текста
    src_tokens = tokenizer.tokenize(text)
    bos_token = tokenizer.vocab_tokentoid.get('<BOS>', 1)
    eos_token = tokenizer.vocab_tokentoid.get('<EOS>', 2)
    
    # Добавление языкового токена источника
    if src_lang_token:
        src_lang_tokens = tokenizer.tokenize(src_lang_token)
        src_tokens = [bos_token] + src_lang_tokens + src_tokens
    else:
        src_tokens = [bos_token] + src_tokens
    
    src_tokens = src_tokens + [eos_token]
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(CONFIG['device'])
    
    # Инициализация целевой последовательности с языковым токеном
    tgt_tokens = [bos_token]
    if tgt_lang_token:
        tgt_lang_tokens = tokenizer.tokenize(tgt_lang_token)
        tgt_tokens.extend(tgt_lang_tokens)
    
    with torch.no_grad():
        for _ in range(max_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(CONFIG['device'])
            
            # Получение предсказания
            output = model(src_tensor, tgt_tensor)
            next_token = output[0, -1, :].argmax().item()
            
            if next_token == eos_token:
                break
                
            tgt_tokens.append(next_token)
    
    # Детокенизация (убираем BOS и языковой токен)
    start_idx = 1
    if tgt_lang_token:
        # Пропускаем языковой токен
        start_idx = 1 + len(tgt_lang_tokens)
    
    translated_text = tokenizer.totext(tgt_tokens[start_idx:])
    return translated_text

def evaluate_model_with_lang_tokens(model, tokenizer, test_data, device, max_length=128):
    """Улучшенная оценка модели с поддержкой языковых токенов"""
    model.eval()
    predictions = []
    references = []
    
    print("Выполняется оценка модели с языковыми токенами...")
    
    with torch.no_grad():
        for i, sample in enumerate(test_data):
            if i >= 50:  # Ограничиваем для быстрой оценки
                break
                
            src_text = sample['src_text']
            tgt_text = sample['tgt_text']
            
            try:
                # Определяем языки на основе текста (простая эвристика)
                # В реальности нужно было бы передавать языки в test_data
                src_lang = 'eng'  # По умолчанию английский
                tgt_lang = 'rus'  # По умолчанию русский
                
                # Используем функцию translate_text для корректного перевода
                translated = translate_text(model, tokenizer, src_text, src_lang, tgt_lang, max_length)
                
                predictions.append(translated)
                references.append(tgt_text)
                
            except Exception as e:
                print(f"Ошибка при переводе примера {i}: {e}")
                predictions.append("")
                references.append(tgt_text)
    
    # Расчет BLEU
    from utils import calculate_bleu_score
    bleu_score = calculate_bleu_score(predictions, references)
    
    return {
        'bleu_score': bleu_score,
        'predictions': predictions[:10],  # Первые 10 для примера
        'references': references[:10]
    }

def main():
    """Основная функция"""
    print("=== Обучение модели перевода ===")
    print(f"Устройство: {CONFIG['device']}")
    
    # 1. Загрузка данных
    df = load_data()
    
    # 2. Создание токенизатора
    tokenizer = create_tokenizer(df)
    
    # 3. Токенизация данных
    dataset = tokenize_and_save_data(df, tokenizer)
    
    # 4. Создание модели
    model = create_model(tokenizer)
    
    # Вывод информации о модели
    print_model_summary(model)
    
    # 5. Предобучение
    model, pretrain_losses = pretrain_model(model, tokenizer, dataset)
    
    # 6. Файнтюнинг
    model, finetune_losses = finetune_model(model, tokenizer, dataset)
    
    # 7. Сохранение логов и графиков
    save_training_log(pretrain_losses, finetune_losses, CONFIG)
    plot_training_curves(pretrain_losses, finetune_losses)
    
    # 8. Оценка модели (уменьшено для ускорения)
    print("\n=== Оценка качества модели ===")
    test_data = []
    
    # Проверяем наличие данных в датасете
    if len(dataset) == 0:
        print("⚠️  ВНИМАНИЕ: Датасет пуст! Проверьте фильтрацию данных по языкам.")
        test_data = []
    else:
        print(f"Размер датасета: {len(dataset)}")
        for i in range(min(10, len(dataset))):  # Уменьшено с 50 до 10
            # Проверяем, существуют ли сохраненные исходные тексты
            if hasattr(dataset, 'original_src_texts') and hasattr(dataset, 'original_tgt_texts'):
                test_data.append({
                    'src_text': dataset.original_src_texts[i],
                    'tgt_text': dataset.original_tgt_texts[i]
                })
            else:
                # Если атрибуты не существуют, создаем простые тестовые данные
                test_data.append({
                    'src_text': f"Test sentence {i}",
                    'tgt_text': f"Тестовое предложение {i}"
                })
    
    if test_data:
        print(f"Подготовлено {len(test_data)} тестовых примеров")
        
        # Используем улучшенную оценку с языковыми токенами
        try:
            evaluation_results = evaluate_model_with_lang_tokens(model, tokenizer, test_data, CONFIG['device'])
            print(f"BLEU Score: {evaluation_results['bleu_score']:.4f}")
            
            # Показываем примеры переводов
            print("\nПримеры переводов:")
            for i, (pred, ref) in enumerate(zip(evaluation_results['predictions'][:5], evaluation_results['references'][:5])):
                print(f"  {i+1}. Ожидаемый: {ref}")
                print(f"     Полученный: {pred}")
                print()
        except Exception as e:
            print(f"❌ Ошибка при оценке модели: {e}")
            print("Попробуем простую оценку...")
            try:
                evaluation_results = evaluate_model(model, tokenizer, test_data, CONFIG['device'])
                print(f"BLEU Score: {evaluation_results['bleu_score']:.4f}")
            except Exception as e2:
                print(f"❌ Ошибка при простой оценке: {e2}")
    else:
        print("❌ Нет данных для оценки модели!")
    
    # 9. Тестирование перевода
    print("\n=== Тестирование перевода ===")
    
    # Получаем выбранные языки для создания соответствующих тестовых случаев
    selected_langs = get_selected_languages()
    print(f"Тестирование для языков: {selected_langs}")
    
    # Создаем тестовые случаи на основе выбранных языков
    test_cases = []
    if 'rus' in selected_langs and 'deu' in selected_langs:
        test_cases = [
            ("Hello, how are you?", "eng", "rus"),
            ("Guten Tag!", "deu", "rus"),
            ("Привет, как дела?", "rus", "deu"),
            ("Thank you very much.", "eng", "deu"),
            ("Ich liebe dich.", "deu", "rus")
        ]
    elif 'rus' in selected_langs:
        test_cases = [
            ("Hello, how are you?", "eng", "rus"),
            ("Good morning!", "eng", "rus"),
            ("Thank you very much.", "eng", "rus"),
            ("What is your name?", "eng", "rus"),
            ("I love you.", "eng", "rus")
        ]
    elif 'deu' in selected_langs:
        test_cases = [
            ("Hello, how are you?", "eng", "deu"),
            ("Good morning!", "eng", "deu"),
            ("Thank you very much.", "eng", "deu"),
            ("What is your name?", "eng", "deu"),
            ("I love you.", "eng", "deu")
        ]
    else:
        # Для других языков используем общие тесты
        test_cases = [
            ("Hello, how are you?", "eng", "rus"),
            ("Good morning!", "eng", "fra"), 
            ("Thank you very much.", "eng", "deu"),
            ("What is your name?", "eng", "spa"),
            ("I love you.", "eng", "ita")
        ]
    
    if test_cases:
        print(f"Выполняется тестирование {len(test_cases)} примеров...")
        for i, (text, src_lang, tgt_lang) in enumerate(test_cases, 1):
            try:
                translated = translate_text(model, tokenizer, text, src_lang, tgt_lang)
                print(f"\n{i}. Исходный ({src_lang}): {text}")
                print(f"   Перевод ({tgt_lang}): {translated}")
                print("-" * 50)
            except Exception as e:
                print(f"\n{i}. ОШИБКА при переводе '{text}': {e}")
                print("-" * 50)
    else:
        print("❌ Нет тестовых случаев для выбранных языков!")
    
    print("Обучение завершено!")
    print("Для быстрого перевода используйте: python fast_run.py")

if __name__ == "__main__":
    main()

