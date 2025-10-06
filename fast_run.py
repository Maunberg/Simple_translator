#!/usr/bin/env python3
"""
Быстрый запуск модели перевода
Использование: python fast_run.py
"""

import torch
import json
import os
from model import Transformer
from tokenizer import Tokeniser


model_path = "/mnt/asr_hot/dutov/study/ml/lab_1_nlp/final_model.pth"
config_path = "/mnt/asr_hot/dutov/study/ml/lab_1_nlp/training_log.json"
test_cases = [
            ("Привет. Как дела?", "rus", "deu"),
            ("Hallo, wie geht's?", "deu", "rus"),
            ("Я тебя люблю!", "rus", "deu"),
            ("Ich liebe dich.", "deu", "rus"),
        ]


class FastTranslator:
    def __init__(self, model_path='final_model.pth', config_path='model_config.json', vocab_path='vocab.json'):
        """Инициализация переводчика"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Используется устройство: {self.device}")
        
        # Загрузка конфигурации
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)['config']
        else:
            # Конфигурация по умолчанию
            self.config = {
                'd_model': 512,
                'num_heads': 8,
                'num_layers': 6,
                'd_ff': 2048,
                'max_seq_length': 128,
                'dropout': 0.1,
                'device': self.device
            }
        
        # Загрузка токенизатора
        self.tokenizer = Tokeniser()
        if os.path.exists(vocab_path):
            self.tokenizer.load_vocab(vocab_path)
            print(f"Загружен словарь размером: {len(self.tokenizer)}")
        else:
            raise FileNotFoundError(f"Словарь не найден: {vocab_path}")
        
        # Загрузка модели
        self.model = self._load_model(model_path)
        print("Модель загружена успешно!")
    
    def _load_model(self, model_path):
        """Загрузка модели"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        # Создание модели
        model = Transformer(
            src_vocab_size=len(self.tokenizer),
            tgt_vocab_size=len(self.tokenizer),
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            d_ff=self.config['d_ff'],
            max_seq_length=self.config['max_seq_length'],
            dropout=self.config['dropout'],
            device=self.device
        )
        
        # Загрузка весов
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def translate(self, text, src_lang='eng', tgt_lang='rus', max_length=300):
        """Перевод текста с указанием языков"""
        if not text.strip():
            return ""
        
        # Получение языковых токенов через токенизатор
        src_lang_token = self.tokenizer.get_lang_token(src_lang)
        tgt_lang_token = self.tokenizer.get_lang_token(tgt_lang)
        
        # Токенизация входного текста
        src_tokens = self.tokenizer.tokenize(text)
        bos_token = self.tokenizer.vocab_tokentoid.get('<BOS>', 1)
        eos_token = self.tokenizer.vocab_tokentoid.get('<EOS>', 2)
        
        # Добавление языкового токена источника
        if src_lang_token:
            src_lang_tokens = self.tokenizer.tokenize(src_lang_token)
            src_tokens = [bos_token] + src_lang_tokens + src_tokens
        else:
            src_tokens = [bos_token] + src_tokens
        
        src_tokens = src_tokens + [eos_token]
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
        
        # Инициализация целевой последовательности с языковым токеном
        tgt_tokens = [bos_token]
        if tgt_lang_token:
            tgt_lang_tokens = self.tokenizer.tokenize(tgt_lang_token)
            tgt_tokens.extend(tgt_lang_tokens)
        
        with torch.no_grad():
            for _ in range(max_length):
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(self.device)
                
                # Получение предсказания
                output = self.model(src_tensor, tgt_tensor)
                next_token = output[0, -1, :].argmax().item()
                
                if next_token == eos_token:
                    break
                    
                tgt_tokens.append(next_token)
        
        # Детокенизация (убираем BOS и языковой токен)
        try:
            start_idx = 1
            if tgt_lang_token:
                # Пропускаем языковой токен
                start_idx = 1 + len(tgt_lang_tokens)
            
            translated_text = self.tokenizer.totext(tgt_tokens[start_idx:])
            return translated_text.strip()
        except Exception as e:
            print(f"Ошибка при детокенизации: {e}")
            return "Ошибка перевода"
    
    def translate_batch(self, texts, max_length=300):
        """Перевод списка текстов"""
        results = []
        for text in texts:
            translated = self.translate(text, max_length)
            results.append(translated)
        return results

def interactive_translation(model_path, config_path):
    """Интерактивный режим перевода"""
    print("=== Интерактивный переводчик ===")
    print("Доступные языки: eng, rus, fra, deu, spa, ita, por, tur, jpn, epo")
    print("Введите текст для перевода (или 'quit' для выхода)")
    
    try:
        translator = FastTranslator(model_path=model_path, config_path=config_path)
        
        # Выбор языков
        print("\nВыберите языки перевода:")
        src_lang = input("Исходный язык (по умолчанию eng): ").strip() or 'eng'
        tgt_lang = input("Целевой язык (по умолчанию rus): ").strip() or 'rus'
        
        while True:
            text = input(f"\nВведите текст на {src_lang} для перевода на {tgt_lang}: ").strip()
            
            if text.lower() in ['quit', 'exit', 'выход']:
                print("До свидания!")
                break
            
            if not text:
                continue
            
            print("Переводим...")
            translated = translator.translate(text, src_lang, tgt_lang)
            print(f"Перевод: {translated}")
            
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что модель обучена и файлы существуют:")
        print("- final_model.pth (или другая модель)")
        print("- model_config.json")
        print("- vocab.json")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def batch_translation(model_path, config_path, test_cases):
    """Пакетный режим перевода"""
    print("=== Пакетный переводчик ===")
    
    try:
        translator = FastTranslator(model_path=model_path, config_path=config_path)
        print("Переводим тестовые предложения...")
        
        print("\nРезультаты перевода:")
        print("=" * 80)
        for text, src_lang, tgt_lang in test_cases:
            translated = translator.translate(text, src_lang, tgt_lang)
            print(f"Исходный ({src_lang}): {text}")
            print(f"Перевод ({tgt_lang}): {translated}")
            print("-" * 60)
            
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что модель обучена и файлы существуют.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def main():
    """Главная функция"""
    print("Выберите режим работы:")
    print("1. Интерактивный перевод")
    print("2. Пакетный перевод тестовых примеров")
    print("3. Выход")
    
    while True:
        choice = input("\nВведите номер (1-3): ").strip()
        
        if choice == '1':
            interactive_translation(model_path, config_path)
            break
        elif choice == '2':
            batch_translation(model_path, config_path, test_cases)
            break
        elif choice == '3':
            print("До свидания!")
            break
        else:
            print("Пожалуйста, введите 1, 2 или 3")

if __name__ == "__main__":
    main()
