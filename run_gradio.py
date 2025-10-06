#!/usr/bin/env python3
"""
Простой скрипт для запуска Gradio интерфейса
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradio_translator import main

if __name__ == "__main__":
    main()
