#!/usr/bin/env python3
"""
Gradio интерфейс для модели перевода
Использование: python gradio_translator.py
"""

import gradio as gr
import torch
import json
import os
from model import Transformer
from tokenizer import Tokeniser


class GradioTranslator:
    def __init__(self, model_path='final_model_ru_de.pth', config_path='final_ru_de_config.json', vocab_path='vocab.json'):
        """Инициализация переводчика для Gradio"""
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
    
    def translate(self, text, src_lang='rus', tgt_lang='deu', max_length=300):
        """Перевод текста с указанием языков"""
        if not text.strip():
            return ""
        
        try:
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
            start_idx = 1
            if tgt_lang_token:
                # Пропускаем языковой токен
                start_idx = 1 + len(tgt_lang_tokens)
            
            translated_text = self.tokenizer.totext(tgt_tokens[start_idx:])
            return translated_text.strip()
            
        except Exception as e:
            return f"Ошибка перевода: {str(e)}"


def create_gradio_interface():
    """Создание Gradio интерфейса"""
    
    # Инициализация переводчика
    try:
        translator = GradioTranslator(
            model_path="/mnt/asr_hot/dutov/study/ml/lab_1_nlp/final_model_ru_de.pth",
            config_path="/mnt/asr_hot/dutov/study/ml/lab_1_nlp/final_ru_de_config.json",
            vocab_path="/mnt/asr_hot/dutov/study/ml/lab_1_nlp/vocab.json"
        )
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        return None
    
    # Доступные языки
    languages = [
        ("Русский", "rus"),
        ("Немецкий", "deu"),
        ("Английский", "eng"),
        ("Французский", "fra"),
        ("Испанский", "spa"),
        ("Итальянский", "ita"),
        ("Португальский", "por"),
        ("Турецкий", "tur"),
        ("Японский", "jpn"),
        ("Эсперанто", "epo")
    ]
    
    def translate_text(text, source_lang, target_lang):
        """Функция перевода для Gradio"""
        if not text.strip():
            return ""
        
        return translator.translate(text, source_lang, target_lang)
    
    def translate_examples():
        """Примеры для демонстрации"""
        return [
            ["Привет. Как дела?", "rus", "deu"],
            ["Hallo, wie geht's?", "deu", "rus"],
            ["Я тебя люблю!", "rus", "deu"],
            ["Ich liebe dich.", "deu", "rus"],
        ]
    
    def swap_languages(source_lang, target_lang):
        """Функция для обмена языков местами"""
        return target_lang, source_lang
    
    # Создание интерфейса
    with gr.Blocks(
        title="Нейронный Переводчик",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .translate-box {
            min-height: 200px;
        }
        #swap-button {
            margin: 5px 0;
            min-width: 50px;
        }
        #swap-button:hover {
            transform: rotate(180deg);
            transition: transform 0.3s ease;
        }
        #swap-label {
            text-align: center;
            font-size: 12px;
            margin: 5px 0;
        }
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 🌍 Нейронный Переводчик
            Многоязычный переводчик на основе Transformer архитектуры.
            
            **Поддерживаемые языки:** Русский, Немецкий, Английский, Французский, Испанский, Итальянский, Португальский, Турецкий, Японский, Эсперанто
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Входной текст")
                input_text = gr.Textbox(
                    label="Текст для перевода",
                    placeholder="Введите текст для перевода...",
                    lines=5,
                    elem_classes=["translate-box"]
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        source_lang = gr.Dropdown(
                            choices=languages,
                            value="rus",
                            label="Исходный язык",
                            info="Выберите язык исходного текста"
                        )
                    
                    with gr.Column(scale=0, min_width=100):
                        gr.Markdown("**Поменять**<br/>**языки**", elem_id="swap-label")
                        swap_btn = gr.Button(
                            "🔄",
                            size="sm",
                            variant="secondary",
                            elem_id="swap-button"
                        )
                    
                    with gr.Column(scale=1):
                        target_lang = gr.Dropdown(
                            choices=languages,
                            value="deu", 
                            label="Целевой язык",
                            info="Выберите язык перевода"
                        )
                
                translate_btn = gr.Button(
                    "🔄 Перевести",
                    variant="primary",
                    size="lg"
                )
                
                clear_btn = gr.Button(
                    "🗑️ Очистить",
                    variant="secondary"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Результат перевода")
                output_text = gr.Textbox(
                    label="Перевод",
                    lines=5,
                    interactive=False,
                    elem_classes=["translate-box"]
                )
        
        # Примеры
        gr.Markdown("### 💡 Примеры")
        examples = gr.Examples(
            examples=translate_examples(),
            inputs=[input_text, source_lang, target_lang],
            label="Нажмите на пример для быстрого перевода"
        )
        
        # Обработчики событий
        translate_btn.click(
            fn=translate_text,
            inputs=[input_text, source_lang, target_lang],
            outputs=output_text
        )
        
        clear_btn.click(
            fn=lambda: ("", "", ""),
            outputs=[input_text, source_lang, target_lang]
        )
        
        # Обработчик для кнопки обмена языков
        swap_btn.click(
            fn=swap_languages,
            inputs=[source_lang, target_lang],
            outputs=[source_lang, target_lang]
        )
        
        # Автоматический перевод при изменении текста
        input_text.submit(
            fn=translate_text,
            inputs=[input_text, source_lang, target_lang],
            outputs=output_text
        )
        
        # Информация о модели
        with gr.Accordion("ℹ️ Информация о модели", open=False):
            gr.Markdown(
                """
                **Архитектура:** Transformer
                **Устройство:** {device}
                **Размер словаря:** {vocab_size}
                **Максимальная длина последовательности:** {max_length}
                
                **Поддерживаемые направления перевода:**
                - Русский ↔ Немецкий
                - Модель для других комбинаций в процессе обучения
                """.format(
                    device=translator.device,
                    vocab_size=len(translator.tokenizer),
                    max_length=translator.config['max_seq_length']
                )
            )
    
    return interface


def main():
    """Главная функция для запуска Gradio интерфейса"""
    print("Запуск Gradio интерфейса...")
    
    interface = create_gradio_interface()
    if interface is None:
        print("Не удалось создать интерфейс. Проверьте наличие файлов модели.")
        return
    
    # Запуск интерфейса
    interface.launch(
        server_name="0.0.0.0",  # Доступ с любого IP
        server_port=7860,       # Порт
        share=True,            # Не создавать публичную ссылку
        show_error=True,        # Показывать ошибки
        quiet=False,           # Показывать логи
    )


if __name__ == "__main__":
    main()
