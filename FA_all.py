import os

os.environ['NUMBA_THREADING_LAYER'] = 'tbb'
from numba import config

config.THREADING_LAYER = 'tbb'

import numbers
from typing import List, Tuple, Optional, Dict, Any, Union
import gc  # Garbage Collector для кращого управління пам'яттю

import numpy as np
from numba import jit, njit, prange
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Обробка даних і тексту
import re
from string import punctuation
from time import time
import openpyxl

# Dash і візуалізація
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
from dash import callback_context, exceptions
import plotly.graph_objs as go

# Системні і допоміжні бібліотеки
import base64
import io
from os import listdir
import webbrowser
from dash.dependencies import Input, Output, State
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numba
import os

# tkinter for selecting browsing folder
import tkinter as tk
from tkinter import filedialog

# ───────── Globals for Python‐file analysis ─────────
analysis_mode = 'text'
current_model = {}  # token → Ngram
current_tokens = []  # ordered list of tokens
current_windows = []  # list of window‐sizes used
python_metrics = {}  # tok → { dt, fa_vals, fit_vals, R, a, gamma, goodness }
current_L = 0
current_w_s_val = 1
# ─────────────────────────────────────────────────────

import re
from typing import List

import re
from typing import List
from string import punctuation

import os, re
from typing import List


def tokenize_code(data: str) -> List[str]:
    """
    Tokenize code in Python, JS/TS, Java, C/C++:
      - string literals (kept intact, but when you split them out you can re-tokenize as words)
      - identifiers (allows $ for JS)
      - numeric literals
      - multi-char operators (===, !==, >>>, <<=, &&, ||, ++, --, //, /*, */…)
      - single-character punctuation/operators
    """
    token_pattern = re.compile(r"""
        # --- string literals (we match them so they don't break operators) ---
        "(?:\\.|[^"\\])*"           # double-quoted
      | '(?:\\.|[^'\\])*'           # single-quoted
      | (?:\\.|[^\\])*`           # backtick template

        # --- identifiers & keywords ---
      | [A-Za-z_$][\w$]*            # letter/underscore/$ start

        # --- numeric literals ---
      | \d+\.\d+(?:[eE][+-]?\d+)?   # floats
      | \d+(?:[eE][+-]?\d+)?        # ints

        # --- multi-char operators & comment markers ---
      | ===|!==|>>>|>>=|<<=|>>|<<   # equality & shifts
      | &&|\|\||\+\+|--             # logical and inc/dec
      | \+=|-=|\*=|/=|%=            # assignment variants
      | ==|!=|<=|>=|=>              # comparisons & arrow
      | \.\.\.                      # triple-dot
      | //                          # single-line comment start
      | /\*|\*/                     # block-comment delimiters

        # --- single-char operators / punctuation ---
      | [+\-*/%&|\^~!<>=?:;.,(){}$begin:math:display$$end:math:display$]
    """, re.VERBOSE)

    return token_pattern.findall(data)


def tokenize_mixed_content(text: str, filename: str) -> List[str]:
    """
    Split text into comment vs code spans, then:
      - comments        → full code-tokenization (so you get //, ===, words, etc.)
      - code spans      → further split out string literals vs code
                            * string literals → natural-language words
                            * code           → tokenize_code()
      - everything else → natural-language words
    """
    _, ext = os.path.splitext(filename.lower())
    tokens: List[str] = []

    def nat_words(s: str):
        # your existing word-splitter
        return remove_punctuation_for_words(s)

    # PYTHON
    if ext == '.py':
        parts = re.split(r'(\#.*?$|\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\')',
                         text, flags=re.MULTILINE)
        for span in parts:
            if not span:
                continue
            if span.startswith('#') or span.startswith('"""') or span.startswith("'''"):
                # still treat Python comments/docstrings as pure code-tokens
                tokens.extend(tokenize_code(span))
            else:
                # inside code, pull out string literals …
                sub = re.split(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|(?:\\.|[^\\])*)',
                               span, flags=re.DOTALL)
                for ss in sub:
                    if not ss:
                        continue
                    if (ss.startswith('"') and ss.endswith('"')) \
                            or (ss.startswith("'") and ss.endswith("'")) \
                            or (ss.startswith('') and ss.endswith('`')):
                        # natural text inside quotes/backticks
                        tokens.extend(nat_words(ss[1:-1]))
                    else:
                        tokens.extend(tokenize_code(ss))

    # C-STYLE (JS/TS/Java/C/C++)
    elif ext in {'.js', '.ts', '.java', '.c', '.cpp'}:
        parts = re.split(r'(//.*?$|/\*[\s\S]*?\*/)',
                         text, flags=re.MULTILINE)
        for span in parts:
            if not span:
                continue
            if span.startswith('//') or span.startswith('/*'):
                # **now** tokenize comments exactly like code
                tokens.extend(tokenize_code(span))
            else:
                # split out string literals
                sub = re.split(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|(?:\\.|[^\\])*)',
                               span, flags=re.DOTALL)
                for ss in sub:
                    if not ss:
                        continue
                    if (ss.startswith('"') and ss.endswith('"')) \
                            or (ss.startswith("'") and ss.endswith("'")) \
                            or (ss.startswith('') and ss.endswith('`')):
                        tokens.extend(nat_words(ss[1:-1]))
                    else:
                        tokens.extend(tokenize_code(ss))

    # EVERYTHING ELSE
    else:
        tokens.extend(nat_words(text))

    return tokens

import os
import pandas as pd

def save_token_to_excel(filename: str, tok: str):
    # make sure the folder exists
    out_dir = "saved_data"
    os.makedirs(out_dir, exist_ok=True)

    met = python_metrics.get(tok, {})
    df_fluc = pd.DataFrame({
        "w": current_windows,
        "∆F":    met.get("fa_vals", []),
        "fit":   met.get("fit_vals", [])
    })

    base, _ = os.path.splitext(filename)
    fn = f"{base}_{tok}.xlsx"
    out_path = get_unique_path(os.path.join(out_dir, fn))

    with pd.ExcelWriter(out_path) as writer:
        df_fluc.to_excel(writer, sheet_name="fluctuation",  index=False)

def get_unique_path(path: str) -> str:
    """
    If `path` exists, append (1), (2), … before the extension until it's unique.
    """
    base, ext = os.path.splitext(path)
    counter = 1
    candidate = path
    while os.path.exists(candidate):
        candidate = f"{base}({counter}){ext}"
        counter += 1
    return candidate

# Функція для очищення пам'яті
def clear_memory(keep: List[str] = []):
    """
    Очищує пам'ять від великих структур даних, які більше не потрібні.

    Args:
        keep: Список назв змінних, які потрібно зберегти
    """
    global model, df, new_ngram, data, uploaded_files, file_lengths, batch_results

    # Зберігаємо лише необхідні дані для таблиці
    variables_to_keep = keep + ['uploaded_files', 'file_lengths', 'batch_results']

    # Очищення великих глобальних структур даних
    if 'model' not in variables_to_keep and 'model' in globals():
        if isinstance(model, dict):
            model.clear()
        model = {}

    # Очищення DataFrame
    if 'df' not in variables_to_keep and 'df' in globals() and df is not None:
        df = None

    # Очищення даних тексту
    if 'data' not in variables_to_keep and 'data' in globals() and data is not None:
        data = None

    # Очищення об'єкта newNgram
    if 'new_ngram' not in variables_to_keep and 'new_ngram' in globals() and new_ngram is not None:
        new_ngram = None

    # Очищення кешу мемоізованих функцій
    if hasattr(prepare_data, 'clear_cache') and 'prepare_data_cache' not in variables_to_keep:
        prepare_data.clear_cache()

    if hasattr(make_markov_chain, 'clear_cache') and 'make_markov_chain_cache' not in variables_to_keep:
        make_markov_chain.clear_cache()

    # Додаємо агресивне очищення пам'яті за допомогою Python gc
    import gc
    gc.collect(generation=2)  # Запуск повного збирання сміття
    gc.collect(generation=1)
    gc.collect(generation=0)


# Кешування для покращення продуктивності
def memoize(func):
    """
    Декоратор для кешування результатів функцій, щоб уникнути повторних обчислень.
    """
    cache = {}

    def wrapper(*args, **kwargs):
        # Створюємо унікальний ключ на основі аргументів
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    # Додаємо функцію для очищення кешу
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper


def remove_punctuation_for_words(data):
    """
    Розбиває текст на слова та видаляє знаки пунктуації.

    Args:
        data: Вхідний текст

    Returns:
        List[str]: Список оброблених слів
    """
    # Використовуємо ефективніший регулярний вираз один раз
    words = re.findall(r'\b[a-zA-Z0-9]+(?:[-\'][a-zA-Z0-9]+)*\b', data.lower())

    # Обробляємо слова з дефісами та апострофами
    result = []
    for word in words:
        if '-' in word or '\'' in word:
            # Розділяємо слово на підчастини за спеціальними символами
            parts = re.split(r'[-\']', word)
            # Додаємо лише непорожні частини
            result.extend([part for part in parts if part])
        else:
            result.append(word)

    return result


def remove_punctuation(data):
    """
    Видаляє знаки пунктуації з тексту.
    """
    temp = []
    for i in range(len(data)):
        if data[i] in punctuation:
            continue
        else:
            temp.append(data[i].lower())
    return "".join(temp)


toast_visible = False
error_visible = False
analyze_visible = False


class Ngram(dict):
    def __init__(self, iterable=None):  # Ініціалізували наш розподіл як новий об'єкт класу, додаємо наявні елементи
        super(Ngram, self).__init__()
        self.fa = {}
        self.counts = {}
        self.sums = {}
        if iterable:
            self.update(iterable)

    def update(self, iterable):  # Оновлюємо розподіл елементами з наявного ітеруємого набору даних
        for item in iterable:
            if item in self:
                self[item] += 1
            else:
                self[item] = 1

    def hist(self):
        plt.bar(self.keys(), self.values())
        plt.show()


def make_dataframe(model, fmin=3):
    """
    Створює DataFrame для відображення результатів аналізу.

    Args:
        model: Словник моделі з n-грамами
        fmin: Мінімальна частота для включення n-грами в аналіз

    Returns:
        pd.DataFrame: DataFrame з результатами
    """
    # Фільтруємо n-грами за мінімальною частотою
    filtered_data = list(
        filter(lambda x: sum(value for value in model[x].values() if isinstance(value, int)) >= fmin, model))

    # Додаємо new_ngram, якщо вона існує в моделі
    if 'new_ngram' not in filtered_data and 'new_ngram' in model:
        filtered_data.append("new_ngram")

    # Створюємо структуру даних для DataFrame
    data = {"ngram": [],
            "F": np.empty(len(filtered_data), dtype=np.dtype(int))}

    # Заповнюємо дані
    for i, ngram in enumerate(filtered_data):
        data["ngram"].append(ngram)

        if ngram == "new_ngram" and hasattr(model[ngram], 'bool'):
            data['F'][i] = sum(model[ngram].bool)
        elif ngram == "new_ngram":
            # Якщо атрибут bool відсутній, встановлюємо значення за замовчуванням
            data['F'][i] = 0
        elif hasattr(model[ngram], 'pos'):
            data["F"][i] = len(model[ngram].pos)
        else:
            data["F"][i] = 0

    # Створюємо DataFrame з даних
    dffff = pd.DataFrame(data=data)
    return dffff


@memoize
def make_markov_chain(data: List, order: int = 1) -> Dict[str, Ngram]:
    """
    Створює ланцюг Маркова з вхідних даних.

    Args:
        data: Список елементів для побудови ланцюга Маркова
        order: Порядок ланцюга Маркова (кількість попередніх елементів для прогнозу)

    Returns:
        Dict[str, Ngram]: Модель ланцюга Маркова у вигляді словника n-грам
    """
    global model, L, V

    # Створюємо новий словник моделі
    model = dict()
    L = len(data) - order

    # Ініціалізуємо спеціальну n-граму для нових елементів
    model['new_ngram'] = Ngram()
    model['new_ngram'].bool = np.zeros(L, dtype=np.uint8)  # використовуємо uint8 для зменшення пам'яті
    model['new_ngram'].pos = []

    # Використовуємо більш ефективний алгоритм для побудови ланцюга Маркова
    if order > 1:
        for i in range(L - 1):
            window = tuple(data[i: i + order])  # Додаємо в словник

            if window in model:  # Приєднуємо до вже існуючого розподілу
                model[window].update([data[i + order]])
                model[window].pos.append(i + 1)
                model[window].bool[i] = 1
            else:
                model[window] = Ngram([data[i + order]])
                model[window].pos = []
                model[window].pos.append(i + 1)
                model[window].bool = np.zeros(L, dtype=np.uint8)
                model[window].bool[i] = 1
                model['new_ngram'].bool[i] = 1
                model['new_ngram'].pos.append(i + 1)
    else:
        # Попередньо визначаємо множину унікальних елементів для оптимізації
        unique_items = set(data)

        # Ініціалізуємо модель для кожного унікального елемента
        for item in unique_items:
            model[item] = Ngram()
            model[item].pos = []
            model[item].bool = np.zeros(L, dtype=np.uint8)

        # Заповнюємо модель
        for i in range(L):
            item = data[i]
            next_item = data[i + order]

            model[item].update([next_item])
            model[item].pos.append(i + order)
            model[item].bool[i] = 1

            if i == 0:  # Перший елемент
                model['new_ngram'].bool[i] = 1
                model['new_ngram'].pos.append(i + order)

        # З'єднуємо останнє слово з першим та перше з останнім
        model[data[L]].update([data[0]])
        if data[L] not in model[data[L]].pos:
            model[data[L]].pos.append(L + order)
            model[data[L]].bool = np.zeros(L, dtype=np.uint8)
            model[data[L]].bool[L - 1] = 1

        model[data[0]].update([data[L]])

    V = len(model)
    return model


def calculate_distance(positions: np.ndarray, L: int, option: str, ngram: str, min_dist: int = 1) -> np.ndarray:
    """
    Розраховує відстані між позиціями елементів з урахуванням граничних умов.

    Оптимізована для роботи з великими наборами даних за допомогою паралельної обробки.

    Args:
        positions: Масив позицій елементів
        L: Довжина тексту
        option: Тип граничних умов ("no", "ordinary", "periodic")
        ngram: Назва n-грами
        min_dist: Мінімальна відстань (0 або 1)

    Returns:
        np.ndarray: Масив відстаней між елементами
    """
    # Оптимізуємо обробку масиву позицій
    positions = np.array(positions, dtype=np.int32)

    # Переконуємося, що min_dist є цілим числом
    if not isinstance(min_dist, int):
        try:
            min_dist = int(min_dist)
        except (ValueError, TypeError):
            print(f"Warning: min_dist '{min_dist}' is not an integer. Using default min_dist=1")
            min_dist = 1

    # Використовуємо оптимізовані функції відповідно до граничних умов
    if option == "no":
        distances = nbc(positions, L, min_dist)
    elif option == "periodic":
        distances = pbc(positions, L, min_dist)
    else:  # "ordinary"
        distances = obc(positions, L, min_dist)

    return distances


@njit
def nbc(pos, L, min_dist=1):
    """
    Обчислює відстані без граничних умов.

    Оптимізовано за допомогою Numba JIT з паралельною обробкою.

    Args:
        pos: Масив позицій елементів
        L: Довжина послідовності
        min_dist: Мінімальна відстань

    Returns:
        np.ndarray: Масив відстаней
    """
    n = len(pos)
    dt = np.zeros(n - 1, dtype=np.int32)

    for i in prange(n - 1):
        dt[i] = pos[i + 1] - pos[i]
        if min_dist == 0:
            dt[i] -= 1

    return dt


@njit
def pbc(pos, L, min_dist=1):
    """
    Обчислює відстані з періодичними граничними умовами.

    Оптимізовано за допомогою Numba JIT з паралельною обробкою.

    Args:
        pos: Масив позицій елементів
        L: Довжина послідовності
        min_dist: Мінімальна відстань

    Returns:
        np.ndarray: Масив відстаней
    """
    n = len(pos)
    dt = np.zeros(n, dtype=np.int32)

    for i in prange(n - 1):
        dt[i] = pos[i + 1] - pos[i]
        if dt[i] > L // 2:
            dt[i] = L - dt[i]
        if min_dist == 0:
            dt[i] -= -1

    # Останній елемент обчислюємо окремо через періодичність
    dt[n - 1] = L - pos[n - 1] + pos[0]
    if dt[n - 1] > L // 2:
        dt[n - 1] = L - dt[n - 1]
    if min_dist == 0:
        dt[n - 1] -= 1

    return dt


@njit
def obc(pos, L, min_dist=1):
    """
    Обчислює відстані зі звичайними граничними умовами.

    Оптимізовано за допомогою Numba JIT з паралельною обробкою.

    Args:
        pos: Масив позицій елементів
        L: Довжина послідовності
        min_dist: Мінімальна відстань

    Returns:
        np.ndarray: Масив відстаней
    """
    n = len(pos)
    dt = np.zeros(n, dtype=np.int32)

    for i in prange(n - 1):
        dt[i] = pos[i + 1] - pos[i]
        if min_dist == 0:
            dt[i] -= 1

    # Останній елемент обчислюємо окремо
    dt[n - 1] = L - pos[n - 1] + pos[0]
    if dt[n - 1] < min_dist:
        dt[n - 1] = min_dist
    if min_dist == 0:
        dt[n - 1] -= 1

    return dt


@jit(nopython=True)
def s(window: np.ndarray) -> int:
    """
    Обчислює суму значень вікна.

    Args:
        window: Масив значень

    Returns:
        int: Сума значень
    """
    # Використовуємо оптимізовану NumPy функцію
    return np.sum(window)


@njit(fastmath=True)
def mse(x: np.ndarray) -> float:
    """
    Обчислює середньоквадратичну похибку (MSE) набору значень.

    Args:
        x: Масив значень

    Returns:
        float: Значення MSE
    """
    if len(x) == 0:
        return 0.0

    # Оптимізоване обчислення MSE
    mean_x = np.mean(x)
    return np.sqrt(np.mean((x - mean_x) ** 2))


@jit(nopython=True, fastmath=True)
def R(x: np.ndarray) -> float:
    """
    Обчислює коефіцієнт варіації.

    Args:
        x: Масив значень

    Returns:
        float: Значення коефіцієнта варіації
    """
    if len(x) <= 1:
        return 0.0

    # Оптимізоване обчислення коефіцієнта варіації
    mean_x = np.mean(x)
    if mean_x == 0:  # Запобігаємо діленню на нуль
        return 0.0
    std_x = np.std(x)
    return std_x / mean_x


@njit(fastmath=True)
def calc_non_overlapping_shift(k, min_window, window_expansion):
    """
    Розраховує зміщення для режиму non-overlapping
    k - номер кроку (починаючи з 1)
    """
    # Numba не працює з None значеннями, тому перевірка робиться в make_windows
    if k == 1:
        return min_window
    else:
        return min_window + (k - 1) * window_expansion


@njit(fastmath=True)
def make_windows(x: np.ndarray, wi: int, l: int, wsh: int,
                 overlap_mode: str = "overlapping",
                 min_window: Optional[int] = None,
                 window_expansion: Optional[int] = None) -> np.ndarray:
    """
    Створює вікна для аналізу даних.

    Args:
        x: Вхідний масив даних
        wi: Розмір вікна
        l: Довжина даних
        wsh: Величина зсуву вікна
        overlap_mode: Режим перекриття вікон ("overlapping" або "non-overlapping")
        min_window: Мінімальний розмір вікна для режиму non-overlapping
        window_expansion: Значення розширення вікна для режиму non-overlapping

    Returns:
        np.ndarray: Масив сум у вікнах
    """
    # Використовуємо Numba для оптимізації
    if overlap_mode == "overlapping":
        # Визначаємо кількість вікон заздалегідь для уникнення повторного обчислення
        num_windows = (l - wi) // wsh + 1
        sums = np.zeros(num_windows, dtype=np.float64)

        # Використовуємо ефективніший цикл
        for i in range(num_windows):
            start_idx = i * wsh
            end_idx = start_idx + wi
            # Використовуємо вбудовану функцію sum у NumPy
            sums[i] = np.sum(x[start_idx:end_idx])

    else:  # non-overlapping режим
        # Використовуємо правильні значення за замовчуванням
        min_win = wi if min_window is None else min_window
        win_exp = wi if window_expansion is None else window_expansion

        # Визначаємо кількість вікон
        num_windows = (l - wi) // wi + 1
        sums = np.zeros(num_windows, dtype=np.float64)

        # Використовуємо ефективніший цикл для non-overlapping
        for i in range(num_windows):
            start_idx = i * wi
            end_idx = start_idx + wi
            if end_idx > l:
                end_idx = l
            sums[i] = np.sum(x[start_idx:end_idx])

    return sums


@njit(fastmath=True)
def calc_sum(x):
    sums = np.empty(len(x))
    for i, w in enumerate(x):
        sums[i] = np.sum(w)
    return sums


@jit(nopython=True, fastmath=True)
def fit(x, a, b):
    return a * (x ** b)


@memoize
def prepare_data(data: str, n: int, split: str) -> List:
    """
    Підготовка даних для аналізу, розбиття на n-грами залежно від вказаних параметрів.

    Args:
        data: Вхідний текст для обробки
        n: Розмір n-грами
        split: Метод розбиття тексту ("word", "letter", "symbol")

    Returns:
        List: Список підготовлених даних
    """
    global L
    if n is None:
        return dash.no_update

    # Використовуємо спільний код попередньої обробки для всіх типів
    data = re.sub(r'\n+', '\n', data)
    data = re.sub(r'\n\s\s', '\n', data)
    data = re.sub(r'﻿', '', data)

    # Для n=1 (одиничні елементи)
    if n == 1:
        if split == "word":
            # Обробка тексту для слів
            data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor()
            processor.preprocess(data)
            result = processor.get_words()
            L = len(result)
            return result

        elif split == 'letter':
            # Обробка для літер і чисел
            temp = []
            data = remove_punctuation(data)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            L = len(temp)
            return temp

        elif split == 'symbol':
            # Обробка для символів
            result = []
            for char in data:
                if char == " " or char == "\n" or char == "\ufeff":
                    result.append("space")
                else:
                    result.append(char.lower())
            L = len(result)
            return result

    # Для n>1 (n-грами)
    else:
        if split == "word":
            # Обробка для n-грам слів
            data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor()
            processor.preprocess(data)
            words = processor.get_words()
            L = len(words)

            # Створюємо n-грами з слів
            result = []
            for i in range(L - n + 1):
                window = tuple(words[i:i + n])
                result.append(window)

            return result

        elif split == "letter":
            # Обробка для n-грам літер і чисел
            temp = []
            data = remove_punctuation(data)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            L = len(temp)
            data = temp
            temp = []
            for i in range(L - n + 1):
                window = tuple(data[i:i + n])
                temp.append(window)
            return temp

        elif split == 'symbol':
            # Обробка для n-грам символів
            temp = []
            for char in data:
                if char == " " or char == "\n" or char == "\ufeff":
                    temp.append("space")
                else:
                    temp.append(char.lower())
            data = temp
            L = len(data)
            temp = []
            for i in range(L - n + 1):
                window = tuple(data[i:i + n])
                temp.append(window)
            return temp

    return []


def dfa(data: List, args: Tuple[int, int, int],
        overlap_mode: str = "overlapping",
        min_window: Optional[int] = None,
        window_expansion: Optional[int] = None) -> np.ndarray:
    """
    Виконує аналіз флуктуацій (DFA) для даних.

    Args:
        data: Вхідні дані для аналізу
        args: Кортеж (розмір вікна, зсув вікна, довжина даних)
        overlap_mode: Режим перекриття вікон ("overlapping" або "non-overlapping")
        min_window: Мінімальний розмір вікна для режиму non-overlapping
        window_expansion: Значення розширення вікна для режиму non-overlapping

    Returns:
        np.ndarray: Масив результатів DFA аналізу
    """
    wi, wh, l = args

    if overlap_mode == "overlapping":
        # Стандартний режим з фіксованим зміщенням
        window_count = len(range(0, l - wi, wh))
        count = np.zeros(window_count, dtype=np.uint8)

        for index, i in enumerate(range(0, l - wi, wh)):
            temp_v = []
            x = []
            for ngram in data[i:i + wi]:
                if ngram in temp_v:
                    x.append(0)
                else:
                    temp_v.append(ngram)
                    x.append(1)
            count[index] = s(np.array(x, dtype=np.uint8))
    else:
        # Non-overlapping режим
        if min_window is None:
            min_window = wh
        if window_expansion is None:
            window_expansion = wh

        # Оцінюємо кількість і розташування вікон
        k = 1
        i = 0
        window_positions = []
        while i < l - wi:
            window_positions.append(i)
            shift = calc_non_overlapping_shift(k, min_window, window_expansion)
            i += shift
            k += 1

        count = np.zeros(len(window_positions), dtype=np.uint8)
        for index, i in enumerate(window_positions):
            temp_v = []
            x = []
            for ngram in data[i:i + wi]:
                if ngram in temp_v:
                    x.append(0)
                else:
                    temp_v.append(ngram)
                    x.append(1)
            count[index] = s(np.array(x, dtype=np.uint8))

    return count


class newNgram():
    def __init__(self, data, wh, l):
        self.data = data
        self.count = {}
        self.dfa = {}
        self.wh, self.l = wh, l

    def func(self, w, overlap_mode="overlapping", min_window=None, window_expansion=None):
        if overlap_mode == "non-overlapping" and (min_window is None or window_expansion is None):
            min_window = self.wh
            window_expansion = self.wh
        count = dfa(self.data, (w, self.wh, self.l), overlap_mode, min_window, window_expansion)
        self.count[w] = count
        self.dfa[w] = float(mse(count))  # Окремо обчислюємо MSE для count


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dictionary to store uploaded files
uploaded_files = {}
# Dictionary to store file lengths with structure: {filename: {'word': length, 'symbol': length, 'letter': length}}
file_lengths = {}
# List to store batch processing results
batch_results = []

# Removing the corpuses list since we're using file upload now
# corpuses = listdir("corpus/")
colors = {
    "background": "#a1a1a1",
    "text": "#a1a1a1"}

import dash_bootstrap_components as dbc

layout2 = html.Div()

layout1 = html.Div([
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Configuration:", style={"background-color": "#e9f5fe", "fontWeight": "bold"}),
                        dbc.CardBody(
                            [
                                # FILE SECTION
                                html.Div([
                                    html.H6("File Selection",
                                            className="text-primary text-center mb-2",
                                            style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),

                                    html.Label("Upload file:"),
                                    html.Div(
                                        [
                                            # Replace dropdown with Upload component
                                            dcc.Upload(
                                                id='upload-data',
                                                children=html.Div([
                                                    'Drag and Drop or ',
                                                    html.A('Select Files',
                                                           style={'fontWeight': 'bold', 'color': '#007bff'})
                                                ]),
                                                style={
                                                    'width': '100%',
                                                    'height': '60px',
                                                    'lineHeight': '60px',
                                                    'borderWidth': '1px',
                                                    'borderStyle': 'dashed',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center',
                                                    'margin': '10px 0',
                                                    'background': '#fafafa',
                                                    'borderColor': '#007bff'
                                                },
                                                multiple=True
                                            ),
                                            html.Div(id='upload-status'),
                                            # Add dropdown for selecting files
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Select file"),
                                                    dcc.Dropdown(
                                                        id='file-selector',
                                                        options=[],
                                                        placeholder="Select a file to analyze",
                                                        style={"minWidth": "250px", "maxWidth": "100%",
                                                               "whiteSpace": "nowrap", "textOverflow": "ellipsis"}
                                                    )
                                                ],
                                                size="md",
                                                className="mb-3",
                                                style={"marginBottom": "10px"}
                                            ),
                                        ]),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee",
                                          "paddingBottom": "10px"}),

                                # ANALYSIS PARAMETERS SECTION
                                html.Div([
                                    html.H6("Analysis Parameters",
                                            className="text-primary text-center mb-2",
                                            style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),

                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Size of ngram"),
                                            dbc.Input(id="n_size", type="number", value=1,
                                                      style={"font-weight": "bold"})
                                        ],
                                        size="md",
                                        className="mb-2"
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Split by"),
                                            dbc.Select(
                                                id="split",
                                                options=[
                                                    {"label": "word", "value": "word"},
                                                    {"label": "letter&number", "value": "letter"},
                                                    {"label": "symbol", "value": "symbol"},
                                                ],
                                                value="word"
                                            )
                                        ],
                                        size="md",
                                        className="mb-2"
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="condition",
                                                options=[
                                                    {"label": "no", "value": "no"},
                                                    {"label": "periodic", "value": "periodic"},
                                                    {"label": "ordinary", "value": "ordinary"}
                                                ],
                                                value="periodic",
                                                style={"font-weight": "bold"}
                                            ),
                                            dbc.InputGroupText("Boundary Condition:")
                                        ],
                                        size="md",
                                        className="mb-2"
                                    ),
                                    dbc.InputGroup([
                                        dbc.InputGroupText("Min Tau:"),
                                        dbc.Select(
                                            id="min_dist_option",
                                            options=[
                                                {"label": "0", "value": "0"},
                                                {"label": "1", "value": "1"}
                                            ],
                                            value="1",
                                            style={"font-weight": "bold"}
                                        )
                                    ], className="mb-1"),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("filter"),
                                            dbc.Input(id="f_min", type="number", value=3, min=1,
                                                      style={"font-weight": "bold"})
                                        ],
                                        className="mb-3"
                                    ),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee",
                                          "paddingBottom": "10px"}),

                                # WINDOW SETTINGS SECTION
                                html.Div([
                                    html.H6("Sliding Window Settings",
                                            className="text-primary text-center mb-2",
                                            style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),

                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="overlap_mode",
                                                options=[
                                                    {"label": "overlapping", "value": "overlapping"},
                                                    {"label": "non-overlapping", "value": "non-overlapping"}
                                                ],
                                                value="overlapping"
                                            ),
                                            dbc.InputGroupText("Window Mode"),
                                        ], size="md", className="mb-2"
                                    ),

                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="def",
                                                options=[
                                                    {"label": "static", "value": "static"},
                                                    {"label": "dynamic", "value": "dynamic"}
                                                ],
                                                value="static"
                                            ),
                                            dbc.InputGroupText("Definition", style={"background-color": "#e9f5fe"}),
                                            dbc.Tooltip(
                                                "Static: Manual window parameters. Dynamic: Auto-calculated based on data size",
                                                target="def",
                                            ),
                                        ], size="md", className="mb-3"
                                    ),

                                    html.Div([
                                        html.Small([
                                            html.Span("w_min = Min Window", style={"fontWeight": "bold"}), " | ",
                                            html.Span("w_s = Window Shift", style={"fontWeight": "bold"}), " | ",
                                            html.Span("w_e = Window Expansion", style={"fontWeight": "bold"}), " | ",
                                            html.Span("w_max = Max Window", style={"fontWeight": "bold"})
                                        ], className="text-muted mb-2 d-block text-center"),
                                    ], style={"background": "#f0f8ff", "padding": "6px", "borderRadius": "5px",
                                              "marginBottom": "10px"}),

                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Min", html.Br(), "Window"],
                                                                     style={"lineHeight": "1.2",
                                                                            "textAlign": "center"}),
                                                           style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_min", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-2"),

                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Window", html.Br(), "Shift"],
                                                                     style={"lineHeight": "1.2",
                                                                            "textAlign": "center"}),
                                                           style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_s", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-2"),

                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Window", html.Br(), "Expansion"],
                                                                     style={"lineHeight": "1.2",
                                                                            "textAlign": "center"}),
                                                           style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_e", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-2"),

                                    dbc.InputGroup([
                                        dbc.InputGroupText(html.Span(["Max", html.Br(), "Window"],
                                                                     style={"lineHeight": "1.2",
                                                                            "textAlign": "center"}),
                                                           style={"width": "90px", "background-color": "#e9f5fe"}),
                                        dbc.Input(id="w_max", type="number", style={"font-weight": "bold"}),
                                    ], className="mb-3"),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee",
                                          "paddingBottom": "10px"}),

                                # ACTION BUTTONS SECTION
                                html.Div([
                                    html.H6("Actions",
                                            className="text-primary text-center mb-2",
                                            style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),

                                    dbc.Button("Analyze natural text", id="chain_button", color="primary",
                                               className="w-100 mb-2",
                                               style={"fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
                                               disabled=analyze_visible),
                                    dbc.Button("Analyze code", id="analyze_code", color="secondary",
                                               className="w-100 mb-2"),
                                    dbc.Button("Save data", id="save", color="danger",
                                               className="w-100",
                                               style={"fontWeight": "bold", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                                    html.Div(id="temp_seve",
                                             children=[]
                                             ),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee",
                                          "paddingBottom": "10px"}),

                                # BATCH PROCESSING SECTION
                                html.Div([
                                    html.H6("Batch Processing",
                                            className="text-primary text-center mb-2",
                                            style={"background": "#f8f9fa", "padding": "6px", "border-radius": "5px"}),
                                    # Add the min-max info Div here
                                    html.Div(id='min-max-length-info',
                                             style={"marginTop": "5px", "fontSize": "small", "textAlign": "center",
                                                    "marginBottom": "10px"}),

                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Lmin: Fmin1"),
                                            dbc.Input(id="fmin1", type="number", value=3, min=1,
                                                      style={"font-weight": "bold"})
                                        ],
                                        style={'marginBottom': '5px'}
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Lmax: Fmin2"),
                                            dbc.Input(id="fmin2", type="number", value=5, min=1,
                                                      style={"font-weight": "bold"})
                                        ],
                                        style={'marginBottom': '5px'}
                                    ),
                                    # Add batch window settings options
                                    dbc.Collapse(
                                        [
                                            html.H6("Batch Window Settings",
                                                    style={'marginTop': '10px', 'fontSize': '14px'}),
                                            dbc.InputGroup(
                                                [
                                                    dbc.Select(
                                                        id="batch_window_mode",
                                                        options=[
                                                            {"label": "Use UI settings", "value": "ui"},
                                                            {"label": "Auto per file", "value": "auto"},
                                                        ],
                                                        value="auto"
                                                    ),
                                                    dbc.InputGroupText("Window Mode")
                                                ],
                                                style={'marginBottom': '5px'}
                                            ),
                                        ],
                                        id="batch_window_controls",
                                        is_open=True
                                    ),
                                    dbc.Button("Process All Files", id="batch_process", color="success",
                                               className="w-100",
                                               style={'marginBottom': '10px', "fontWeight": "bold",
                                                      "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                                    dbc.Button(
                                        "Process All Code Files",
                                        id="batch_process_code",
                                        color="secondary",
                                        className="w-100 mb-2"
                                    ),
                                    dbc.Button("Save Batch Results", id="save_batch", color="primary",
                                               className="w-100",
                                               style={'marginBottom': '10px', "fontWeight": "bold",
                                                      "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                                    html.Div(id="temp_seve_batch", style={'marginBottom': '10px'}),
                                ], style={"marginBottom": "15px", "borderBottom": "1px solid #eee",
                                          "paddingBottom": "10px"}),

                                html.Div(id="alert", children=[])
                                # html.H6("Boundary Condition:"),
                                # dcc.RadioItems(id='condition',options=[{"label":"no","value":"no"},{"label":"periodic","value":"periodic"},{"label":"ordinary","value":"ordinary"}],value="words"),
                            ]

                        ),

                    ], color="light", style={"margin-left": "0px", "margin-top": "10px", }
                ),
                width={"size": 3, "offset": 0}
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.Tabs(
                                    [
                                        dbc.Tab(label="DataTable", tab_id="data_table",
                                                label_style={"font-weight": "bold"})
                                    ],
                                    id="dataframe",
                                    active_tab="data_table"
                                )

                            ),
                            dbc.CardBody(
                                [
                                    # here table
                                    html.Div(id="box_tab",
                                             style={"display": "none", "height": "400px", "minHeight": "400px"},
                                             children=[dbc.Spinner(dash_table.DataTable(
                                                 id="table",
                                                 columns=[{"name": i, "id": i} for i in
                                                          ['rank', "ngram", "F", "R", "a", "gamma", "goodness"]],
                                                 style_data={'whiteSpace': 'auto', 'height': 'auto'},
                                                 editable=False,
                                                 filter_action="native",
                                                 sort_action="native",
                                                 page_size=50,
                                                 fixed_rows={'headers': True},
                                                 fixed_columns={'headers': True},
                                                 style_cell={'whiteSpace': 'normal',
                                                             'height': 'auto',
                                                             "widht": "auto",
                                                             'textAlign': 'right',
                                                             "fontSize": 15,
                                                             "font-family": "sans-serif"},
                                                 # 'minWidth': 40, 'width': 95, 'maxWidth': 95},
                                                 style_table={"height": "400px", "minWidth": "500px",
                                                              'overflowY': 'auto', "overflowX": "none",
                                                              "minHeight": "400px"},
                                                 style_header={
                                                     'backgroundColor': '#e9f5fe',
                                                     'fontWeight': 'bold',
                                                     'textAlign': 'center'
                                                 },
                                                 style_data_conditional=[
                                                     {
                                                         'if': {'row_index': 'odd'},
                                                         'backgroundColor': '#f9f9f9'
                                                     },
                                                     {
                                                         'if': {'state': 'selected'},
                                                         'backgroundColor': '#deeaff',
                                                         'border': '1px solid #aaa'
                                                     }
                                                 ]
                                             ))]),
                                    html.Div(id="box_chain",
                                             style={"display": "none"},
                                             children=[dbc.Spinner(dcc.Graph(id="chain", style={"height": "400px"}))]),

                                    dbc.CardHeader("Characteristics",
                                                   style={"padding": "5px 20px", "background-color": "#f0f8ff",
                                                          "font-weight": "bold"}),
                                    # here add chars
                                    dbc.CardBody(
                                        dbc.Row([
                                            # NOTE додала вивід 8-ми значень з екселю а також кнопку для копіювання всього
                                            dbc.Col([
                                                html.Div(["Length: "], id="l",
                                                         style={"whiteSpace": "nowrap", "width": "100%",
                                                                "overflow": "hidden", "textOverflow": "ellipsis",
                                                                "fontWeight": "bold", "padding": "3px"}),
                                                html.Div(["Vocabulary: "], id="v",
                                                         style={"fontWeight": "bold", "padding": "3px"}),
                                                html.Div(["Time: "], id="t",
                                                         style={"fontWeight": "bold", "padding": "3px"})

                                            ], width={"size": 5}),
                                            dbc.Col([
                                                html.Div([""], id="new_output1", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output2", n_clicks=0, style={"padding": "3px"}),
                                            ], width={"size": 2}),
                                            dbc.Col([
                                                html.Div([""], id="new_output3", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output4", n_clicks=0, style={"padding": "3px"}),
                                            ], width={"size": 2}),
                                            dbc.Col([
                                                html.Div([""], id="new_output5", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output6", n_clicks=0, style={"padding": "3px"}),
                                            ], width={"size": 2}),
                                            dbc.Col([
                                                html.Div([""], id="new_output7", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="new_output8", n_clicks=0, style={"padding": "3px"}),
                                                html.Div([""], id="copy_all", n_clicks=0,
                                                         style={"fontWeight": "bold", "color": "#007bff",
                                                                "cursor": "pointer", "textDecoration": "underline",
                                                                "padding": "3px"})
                                            ], width={"size": 1}),
                                        ])
                                    ),
                                    # Add batch results table
                                    html.Div([
                                        html.H5("Batch Processing Results", style={'marginTop': '20px'}),
                                        dbc.Spinner(dash_table.DataTable(
                                            id="batch_table",
                                            columns=[
                                                {"name": "No.", "id": "no"},
                                                {"name": "Filename", "id": "filename"},
                                                {"name": "F_min", "id": "f_min"},
                                                {"name": "Length (L)", "id": "length"},
                                                {"name": "Vocabulary (V)", "id": "vocabulary"},
                                                {"name": "Time (s)", "id": "time"},
                                                {"name": "R_avg", "id": "r_avg"},
                                                {"name": "dR", "id": "dr"},
                                                {"name": "Rw_avg", "id": "rw_avg"},
                                                {"name": "dRw", "id": "drw"},
                                                {"name": "gamma_avg", "id": "gamma_avg"},
                                                {"name": "dgamma", "id": "dgamma"},
                                                {"name": "gammaw_avg", "id": "gammaw_avg"},
                                                {"name": "dgammaw", "id": "dgammaw"}
                                            ],
                                            style_data={'whiteSpace': 'normal', 'height': 'auto'},
                                            style_cell={'textAlign': 'center'},
                                            style_header={'fontWeight': 'bold', 'backgroundColor': '#e9f5fe'},
                                            style_table={"overflowX": "auto"},
                                            style_data_conditional=[
                                                {
                                                    'if': {'row_index': 'odd'},
                                                    'backgroundColor': '#f9f9f9'
                                                },
                                                {
                                                    'if': {'row_index': -1},
                                                    'fontWeight': 'bold',
                                                    'backgroundColor': 'lightyellow'
                                                },
                                                {
                                                    'if': {'row_index': -2},
                                                    'fontWeight': 'bold',
                                                    'backgroundColor': 'lightblue'
                                                }
                                            ]
                                        )),
                                        # Remove the old batch save button
                                    ], id="batch_results_container", style={"display": "none"})
                                ]
                            )
                        ], style={"padding": "0", "margin-right": "0px", "margin-top": "10px", "height": "auto",
                                  "minHeight": "650px"}),
                ],
                width={"size": 9, "padding": 0}
            ),
        ]
    ),
    dbc.Row([
        dbc.Col(
            width={"size": 6, "offset": 0},
            children=[
                dbc.Card(
                    [
                        dbc.CardHeader(
                            dbc.Tabs(
                                [
                                    dbc.Tab(label="distribution", tab_id="tab1", label_style={"font-weight": "bold"}),
                                ],
                                id='card-tabs1',
                                active_tab="tab1"
                                # active_tab="tab1",
                                # card=True
                            )
                        ),
                        dbc.CardBody([
                            dcc.Graph(id="graphs", config={'displayModeBar': True, 'displaylogo': False})

                        ], style={"background-color": "#fcfcfc"})
                    ], style={"height": "100%", "widht": "100%", "margin-right": "0%", "margin-top": "10px",
                              "margin-left": "0%"}
                )
            ]),
        dbc.Col(
            width={"size": 6},
            children=[

                dbc.Card(
                    [
                        dbc.CardHeader(
                            dbc.Tabs(
                                [
                                    dbc.Tab(label="fluctuation", tab_id="tab2", label_style={"font-weight": "bold"}),
                                    dbc.Tab(label="gamma/R", tab_id="tab3", label_style={"font-weight": "bold"})
                                ],
                                id='card-tabs',
                                active_tab="tab2"
                                # active_tab="tab2",
                                # card=True
                            )
                        ),
                        dbc.CardBody([
                            dcc.RadioItems(
                                id="scale",
                                options=[
                                    {"label": "linear", "value": "linear"},
                                    {"label": "log", "value": "log"}
                                ],
                                value="linear",
                                labelStyle={"marginRight": "15px", "fontWeight": "bold"},
                                inputStyle={"marginRight": "5px"},
                                style={"marginBottom": "10px", "backgroundColor": "#f8f9fa", "padding": "8px",
                                       "borderRadius": "5px"}
                            ),
                            dcc.Graph(id="fa", config={'displayModeBar': True, 'displaylogo': False})

                        ], style={"background-color": "#fcfcfc"})

                    ], style={"height": "100%", "widht": "100%", "padding": "0", "margin-right": "0%",
                              "margin-top": "10px", "margin-left": "0%"}
                )

            ]
        )
    ]

    ),
    dbc.Row(
        children=[
            html.Br(),
            html.Br()
        ]
    ),
    dcc.Store(id='stored-data'),
    html.Div(id='output-message'),
    dbc.Toast(
        id="click-toast",
        header="Attention",
        icon="danger",
        is_open=error_visible,
        dismissable=True,
        duration=6000,
        children="Length has not been calculated yet!",
        style={"position": "fixed", "top": "40%", "right": "40%", "width": 500, "zIndex": 9999}
    ),
])
from dash.dependencies import Input, Output, State

app.layout = layout1
df = None
g = None
import plotly.express as px
from sklearn.metrics import r2_score
import networkx as nx
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numba
import os


def is_number(s: str) -> bool:
    """
    Перевіряє, чи можна рядок перетворити в число.

    Args:
        s: Рядок для перевірки

    Returns:
        bool: True, якщо рядок може бути перетворений у число, інакше False
    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# NOTE клас із С# для обробки слів
class NgrammProcessor:
    """
    Клас для обробки тексту і отримання n-грам.
    """

    def __init__(self, ignore_punctuation: bool = True):
        """
        Ініціалізує процесор n-грам.

        Args:
            ignore_punctuation: Чи ігнорувати пунктуацію при обробці
        """
        self.ignore_punctuation = ignore_punctuation
        self.words = []
        self.processed_text = ""

    def preprocess(self, text: str) -> None:
        """
        Попередня обробка тексту.

        Args:
            text: Вхідний текст для обробки
        """
        # Видаляємо пунктуацію, якщо потрібно
        if self.ignore_punctuation:
            # Використовуємо оптимізований метод видалення пунктуації
            self.processed_text = ''.join(
                char for char in text if char not in punctuation or char == '-' or char == "'")
        else:
            self.processed_text = text

        # Розбиваємо текст на слова
        self.words = [word.lower() for word in re.findall(r'\b\w+(?:[-\']\w+)*\b', self.processed_text)]

    def get_words(self, remove_empty_entries: bool = False) -> List[str]:
        """
        Отримує список слів із обробленого тексту.

        Args:
            remove_empty_entries: Чи видаляти порожні рядки

        Returns:
            List[str]: Список слів
        """
        if remove_empty_entries:
            return [word for word in self.words if word]
        return self.words


def is_valid_letter(char: str) -> bool:
    """
    Check if a character should be skipped.
    Returns True if character should be skipped.
    """
    invalid_characters = [' ', '\n', '\ufeff', '°', '"', '„', '–']
    return char in invalid_characters


length_updated = False


@app.callback(
    [Output('upload-status', 'children'),
     Output('file-selector', 'options'),
     Output('min-max-length-info', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('n_size', 'value'),
     State('split', 'value')]
)
def update_upload_status(contents, filenames, n_size, split_mode):
    import re, base64
    global uploaded_files, file_lengths

    # prepare return‐values
    min_max_info = ""
    options = [{'label': fn, 'value': fn, 'title': fn}
               for fn in uploaded_files]

    # no new upload: just recompute Min/Max if we already have files
    if not contents:
        if file_lengths and split_mode:
            lengths = [file_lengths[fn].get(split_mode, 0)
                       for fn in file_lengths]
            if lengths:
                split_label = "letters&numbers" if split_mode == 'letter' else f"{split_mode}s"
                min_max_info = f"Min/Max Length ({split_label}): {min(lengths)} / {max(lengths)}"
        return html.Div("No new files uploaded"), options, html.Div(min_max_info)

    success_count = 0
    error_count = 0

    for content, filename in zip(contents, filenames):
        try:
            # decode the upload
            header, b64 = content.split(',', 1)
            raw = base64.b64decode(b64)

            # for .py apply mixed‐encoding (CP1251 for code, UTF-8 for comments/docstrings)
            if filename.lower().endswith('.py'):
                pieces = []
                token_re = re.compile(
                    rb"""
                      (\# [^\n]*           )  # single-line comment
                    | (\"\"\".*?\"\"\"     )  # triple-quoted double
                    | (\'\'\'.*?\'\'\'     )  # triple-quoted single
                    """,
                    re.MULTILINE | re.DOTALL | re.VERBOSE
                )
                last = 0
                for m in token_re.finditer(raw):
                    # code before comment
                    if m.start() > last:
                        pieces.append(raw[last:m.start()]
                                      .decode('cp1251', errors='replace'))
                    # comment/docstring span
                    pieces.append(raw[m.start():m.end()]
                                  .decode('utf-8', errors='replace'))
                    last = m.end()
                # trailing code
                if last < len(raw):
                    pieces.append(raw[last:]
                                  .decode('cp1251', errors='replace'))
                file_content = "".join(pieces)

            else:
                # everything else is natural text, UTF-8
                file_content = raw.decode('utf-8', errors='replace')

            # store
            uploaded_files[filename] = file_content
            file_lengths[filename] = {}

            # now compute lengths exactly as before
            # 1) words
            txt = re.sub(r'\n+', '\n', file_content)
            txt = re.sub(r'\n\s\s', '\n', txt)
            txt = re.sub(r'﻿', '', txt)
            txt = re.sub(r'--', ' -', txt)
            proc = NgrammProcessor()
            proc.preprocess(txt)
            words = proc.get_words()
            file_lengths[filename]['word'] = len(words)

            # 2) symbols
            syms = []
            for ch in file_content:
                if ch in (" ", "\n", "\ufeff"):
                    syms.append("space")
                else:
                    syms.append(ch.lower())
            file_lengths[filename]['symbol'] = len(syms)

            # 3) letters&numbers
            lett = remove_punctuation(file_content)
            letters = []
            for w in lett:
                for c in w:
                    if c != ' ':
                        letters.append(c)
            file_lengths[filename]['letter'] = len(letters)

            success_count += 1

        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            error_count += 1

    # summary message
    summary = html.Div([
        html.H5("Upload Summary:"),
        html.P(f"Successfully uploaded: {success_count}", style={'color': 'green'}),
        html.P(f"Errors: {error_count}", style={'color': 'red' if error_count else 'green'})
    ])

    # rebuild selector options
    options = [{'label': fn, 'value': fn, 'title': fn}
               for fn in uploaded_files]

    # recompute min/max
    if file_lengths and split_mode:
        lengths = [file_lengths[fn].get(split_mode, 0)
                   for fn in file_lengths]
        if lengths:
            split_label = "letters&numbers" if split_mode == 'letter' else f"{split_mode}s"
            min_max_info = f"Min/Max Length ({split_label}): {min(lengths)} / {max(lengths)}"

    return summary, options, html.Div(min_max_info)


# Add callback to handle file selection
@app.callback(
    [Output('l', 'children'),
     Output('w_min', 'value'),
     Output('w_s', 'value'),
     Output('w_e', 'value'),
     Output('w_max', 'value')],
    [Input('file-selector', 'value'),
     Input('split', 'value')],
    [State('def', 'value'),
     State('n_size', 'value')]
)
def process_selected_file(selected_filename, split, definition, n):
    global L, data, length_updated

    if selected_filename is None or selected_filename not in uploaded_files:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    file = uploaded_files[selected_filename]
    length_updated = False

    # Calculate L based on split type (dynamic or static handles data differently)
    if definition == "dynamic":
        data = prepare_data(file, n, split)
        L = len(data)
        w_max = int(L / 10)
        w_min = int(w_max / 10)
    else:
        # Static mode calculation based on selected split
        if split == "letter":
            temp = []
            data = remove_punctuation(file)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            data = temp
            L = len(data)
        elif split == "symbol":
            temp = []
            for char in file:
                if char == " " or char == "\n" or char == "\ufeff":
                    temp.append("space")
                else:
                    temp.append(char.lower())
            data = temp
            L = len(data)
        elif split == "word":
            file = re.sub(r'\n+', '\n', file)
            file = re.sub(r'\n\s\s', '\n', file)
            file = re.sub(r'﻿', '', file)
            file = re.sub(r'--', ' -', file)
            processor = NgrammProcessor()
            processor.preprocess(file)
            data = processor.get_words()
            L = len(data)

        w_max = int(L / 20)
        w_min = int(w_max / 20)
        length_updated = True

    # Format the lengths into a multi-line Div
    length_elements = [html.Strong("Length:")]

    # Get all lengths from the stored dictionary
    lengths = file_lengths[selected_filename]

    # Add each length type on a new line
    if 'word' in lengths:
        length_elements.append(html.Div(f"words: {lengths['word']}"))
    if 'symbol' in lengths:
        length_elements.append(html.Div(f"symbols: {lengths['symbol']}"))
    if 'letter' in lengths:
        length_elements.append(html.Div(f"letters&numbers: {lengths['letter']}"))

    return length_elements, w_min, w_min, w_min, w_max


def remove_empty_strings(arr: List[str]) -> List[str]:
    """
    Видаляє порожні рядки та спеціальні символи з списку.

    Args:
        arr: Список рядків для обробки

    Returns:
        List[str]: Список без порожніх рядків та спеціальних символів
    """
    return [item for item in arr if item and item != '\ufeff']


new_ngram = None

# Add callback for batch processing
from dash import callback_context, exceptions


@app.callback(
    [Output("batch_table", "data"),
     Output("batch_results_container", "style")],
    [Input("batch_process", "n_clicks"),
     Input("batch_process_code", "n_clicks")],
    [State("fmin1", "value"),
     State("fmin2", "value"),
     State("split", "value"),
     State("n_size", "value"),
     State("condition", "value"),
     State("def", "value"),
     State("min_dist_option", "value"),
     State("overlap_mode", "value"),
     State("w_min", "value"),
     State("w_s", "value"),
     State("w_e", "value"),
     State("w_max", "value"),
     State("batch_window_mode", "value")]
)
def process_all_files(text_clicks, code_clicks,
                      fmin1, fmin2, split, n_size, condition, definition,
                      min_dist_option, overlap_mode, w_min, w_s, w_e, w_max,
                      batch_window_mode):
    import os, gc
    from time import time

    # Which button was clicked?
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    # Partition uploaded_files into text vs code sets
    code_exts = {".py", ".js", ".ts", ".java", ".c", ".cpp"}
    all_files = list(uploaded_files.keys())

    if triggered == "batch_process":
        # natural-text batch: skip any code extensions
        file_list = [(fn, uploaded_files[fn])
                     for fn in all_files
                     if os.path.splitext(fn.lower())[1] not in code_exts]
        if text_clicks is None:
            return [], {"display": "none"}

    elif triggered == "batch_process_code":
        # code-only batch: only files with code extensions
        file_list = [(fn, uploaded_files[fn])
                     for fn in all_files
                     if os.path.splitext(fn.lower())[1] in code_exts]
        if code_clicks is None:
            return [], {"display": "none"}

    else:
        # shouldn't happen
        raise dash.exceptions.PreventUpdate

    # nothing to do?
    if not file_list:
        return [], {"display": "none"}

    # compute lmin/lmax on chosen subset
    lengths = [file_lengths[fn][split] for fn, _ in file_list]
    lmin, lmax = min(lengths), max(lengths)

    batch_results.clear()

    # -- loop over selected files --
    for idx, (filename, file_content) in enumerate(file_list, start=1):
        gc.collect()
        L_file = file_lengths[filename][split]
        # linear interpolate f_min
        if lmin == lmax:
            f_min = fmin1
        else:
            f_min = round(fmin1 + (fmin2 - fmin1) * (L_file - lmin) / (lmax - lmin))

        start_time = time()

        # reuse your existing pipeline *verbatim*
        # 1) prepare data list
        if definition == "dynamic":
            data = prepare_data(file_content, n_size, split)
        else:
            # static split: same code you already have
            if split == "letter":
                cleaned = remove_punctuation(file_content)
                data = [ch for w in cleaned for ch in w if ch != " "]
            elif split == "symbol":
                data = ["space" if ch in {" ", "\n", "\ufeff"} else ch.lower()
                        for ch in file_content]
            else:  # word
                txt = re.sub(r'\n+', '\n', file_content)
                txt = re.sub(r'\n\s\s', '\n', txt)
                txt = re.sub(r'﻿', '', txt)
                txt = re.sub(r'--', ' -', txt)
                proc = NgrammProcessor()
                proc.preprocess(txt)
                data = proc.get_words()
        L = len(data)

        # 2) window parameters
        if batch_window_mode == "ui":
            wm_val = int(w_max) if w_max is not None else max(10, L // 20)
            w_val = int(w_s) if w_s is not None else max(1, wm_val // 10)
            wh_val = w_val
            we_val = int(w_e) if w_e is not None else w_val
        else:
            if definition == "dynamic":
                wm_val = max(10, L // 10)
                w_val = max(1, wm_val // 10)
            else:
                wm_val = max(10, L // 20)
                w_val = max(1, wm_val // 20)
            wh_val = w_val
            we_val = w_val

        wm_val, w_val, wh_val, we_val = map(lambda x: max(1, x),
                                            (wm_val, w_val, wh_val, we_val))

        # 3) build frequency model
        local_model = {}
        for pos, gram in enumerate(data):
            if gram not in local_model:
                ng = Ngram()
                ng.pos = []
                local_model[gram] = ng
            local_model[gram].pos.append(pos)

        # 4) apply f_min filter
        valid = [g for g in local_model if len(local_model[g].pos) >= f_min]

        # 5) compute metrics
        temp_R, temp_a, temp_gamma, temp_err = [], [], [], []
        windows = list(range(w_val, wm_val, we_val))
        for gram in valid:
            ng = local_model[gram]
            # boolean array
            ng.bool = np.zeros(L, dtype=np.uint8)
            for p in ng.pos:
                ng.bool[p] = 1

            # distance
            ng.dt = calculate_distance(
                np.array(ng.pos, dtype=np.uint32),
                L, condition, gram, int(min_dist_option)
            )

            # fluctuation
            ff_vals = []
            for w in windows:
                cnts = make_windows(
                    ng.bool, wi=w, l=L, wsh=wh_val,
                    overlap_mode=overlap_mode,
                    min_window=(w_val if overlap_mode != "overlapping" else None),
                    window_expansion=(we_val if overlap_mode != "overlapping" else None)
                )
                ff_vals.append(mse(cnts))
            ng.fa = dict(zip(windows, ff_vals))

            # fit
            try:
                c, _ = curve_fit(fit, windows, ff_vals, method="lm", maxfev=5000)
                a_val, g_val = c[0], c[1]
                fit_vals = [fit(w, *c) for w in windows]
                err = r2_score(ff_vals, fit_vals)
            except:
                a_val = g_val = err = 0.0

            temp_R.append(round(R(ng.dt), 8))
            temp_a.append(round(a_val, 8))
            temp_gamma.append(round(g_val, 8))
            temp_err.append(round(err, 5))

        # 6) build DataFrame row & collect stats
        V = len(valid)
        elapsed = round(time() - start_time, 3)
        batch_results.append({
            "no": idx,
            "filename": filename,
            "f_min": f_min,
            "length": L,
            "vocabulary": V,
            "time": elapsed,
            "r_avg": round(np.mean(temp_R), 8) if temp_R else 0,
            "dr": round(np.std(temp_R), 8) if temp_R else 0,
            "rw_avg": round(np.average(temp_R, weights=np.array(temp_R) / sum(temp_R)), 8) if temp_R else 0,
            "drw": round(np.sqrt(
                np.average((np.array(temp_R) - np.average(temp_R, weights=np.array(temp_R) / sum(temp_R))) ** 2,
                           weights=np.array(temp_R) / sum(temp_R))), 8) if temp_R else 0,
            "gamma_avg": round(np.mean(temp_gamma), 8) if temp_gamma else 0,
            "dgamma": round(np.std(temp_gamma), 8) if temp_gamma else 0,
            "gammaw_avg": round(np.average(temp_gamma, weights=np.array(temp_gamma) / sum(temp_gamma)),
                                8) if temp_gamma else 0,
            "dgammaw": round(np.sqrt(np.average(
                (np.array(temp_gamma) - np.average(temp_gamma, weights=np.array(temp_gamma) / sum(temp_gamma))) ** 2,
                weights=np.array(temp_gamma) / sum(temp_gamma))), 8) if temp_gamma else 0,
        })

        # cleanup
        del data, local_model
        gc.collect()

    # add MEAN/STDDEV rows if any
    if batch_results:
        add_batch_statistics(batch_results)

    return batch_results, {"display": "block"}


def add_batch_statistics(results):
    """
    Adds mean and standard deviation rows to batch results

    Args:
        results: List of batch results to add statistics to
    """
    if not results:
        return

    # Extract only numerical data for statistics
    data_for_stats = []
    numeric_fields = ["length", "vocabulary", "time", "r_avg", "dr", "rw_avg", "drw",
                      "gamma_avg", "dgamma", "gammaw_avg", "dgammaw"]

    for item in results:
        # Skip statistics rows (if this function is called multiple times)
        if item["filename"] in ["MEAN", "STDDEV"]:
            continue

        data_point = {}
        for field in numeric_fields:
            if field in item:
                data_point[field] = item[field]

        data_for_stats.append(data_point)

    # Calculate means
    if not data_for_stats:
        return

    df_stats = pd.DataFrame(data_for_stats)

    # Calculate means
    means = {
        "no": len(results) + 1,
        "filename": "MEAN",
        "f_min": "-",
    }

    # Calculate standard deviations
    stddevs = {
        "no": len(results) + 2,
        "filename": "STDDEV",
        "f_min": "-",
    }

    # Fill in statistics for all numeric fields
    for field in numeric_fields:
        if field in df_stats.columns:
            means[field] = round(df_stats[field].mean(),
                                 8 if field in ["r_avg", "dr", "rw_avg", "drw", "gamma_avg", "dgamma", "gammaw_avg",
                                                "dgammaw"] else
                                 3 if field == "time" else 0)

            stddevs[field] = round(df_stats[field].std(),
                                   8 if field in ["r_avg", "dr", "rw_avg", "drw", "gamma_avg", "dgamma", "gammaw_avg",
                                                  "dgammaw"] else
                                   3 if field == "time" else 0)

    # Remove old statistics rows if present
    results[:] = [r for r in results if r["filename"] not in ["MEAN", "STDDEV"]]

    # Add statistics to results
    results.append(means)
    results.append(stddevs)


# Update the batch results table to show window parameters too
@app.callback(
    Output("batch_table", "columns"),
    [Input("batch_process", "n_clicks")]
)
def update_batch_table_columns(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    columns = [
        {"name": "No.", "id": "no"},
        {"name": "Filename", "id": "filename"},
        {"name": "F_min", "id": "f_min"},
        {"name": "Length (L)", "id": "length"},
        {"name": "Vocabulary (V)", "id": "vocabulary"},
        {"name": "Time (s)", "id": "time"},
        {"name": "R_avg", "id": "r_avg"},
        {"name": "dR", "id": "dr"},
        {"name": "Rw_avg", "id": "rw_avg"},
        {"name": "dRw", "id": "drw"},
        {"name": "gamma_avg", "id": "gamma_avg"},
        {"name": "dgamma", "id": "dgamma"},
        {"name": "gammaw_avg", "id": "gammaw_avg"},
        {"name": "dgammaw", "id": "dgammaw"}
    ]

    return columns


# Add callback to save batch results
@app.callback(
    Output("temp_seve_batch", "children"),
    [Input("save_batch", "n_clicks")],
    [State("n_size", "value"),
     State("split", "value"),
     State("condition", "value"),
     State("def", "value"),
     State("min_dist_option", "value"),
     State("overlap_mode", "value"),
     State("batch_window_mode", "value")]
)
def save_batch_results(n_clicks, n_size, split, condition, definition,
                       min_dist_option, overlap_mode, batch_window_mode):
    if n_clicks is None:
        return dash.no_update
    if not batch_results:
        return html.Div(["No batch results to save"])
    try:
        # always save into ./saved_data/
        save_folder = "saved_data"
        os.makedirs(save_folder, exist_ok=True)

        df_batch = pd.DataFrame(batch_results)
        base = (f"batch_results_n={n_size},split={split},condition={condition},"
                f"definition={definition},min_dist={min_dist_option},"
                f"overlap={overlap_mode},window_mode={batch_window_mode}")
        filename = os.path.join(save_folder, f"{base}.xlsx")
        unique = get_unique_path(filename)

        df_batch.to_excel(unique, index=False)
        return html.Div([f"Saved!"])
    except Exception as e:
        return html.Div([f"Error saving batch results: {e}"])


@app.callback(
    [
        Output("table", "data"),
        Output("chain", "figure"),
        Output("box_tab", "style"),
        Output("box_chain", "style"),
        Output("alert", "children"),
        Output("v", "children"),
        Output("t", "children"),
        Output("click-toast", "is_open"),
    ],
    [
        Input("chain_button", "n_clicks"),
        Input("analyze_code", "n_clicks"),
        Input("dataframe", "active_tab"),
    ],
    [
        State("file-selector", "value"),
        State("f_min", "value"),
        State("w_min", "value"),
        State("w_s", "value"),
        State("w_e", "value"),
        State("w_max", "value"),
        State("def", "value"),
        State("min_dist_option", "value"),
        State("overlap_mode", "value"),
        State("n_size", "value"),
        State("split", "value"),
        State("condition", "value"),
    ],
)
def update_table(
    chain_clicks,
    code_clicks,
    dataframe,
    filename,
    f_min,
    w_min,
    w_s,
    w_e,
    w_max,
    definition,
    min_dist_option,
    overlap_mode,
    n_size,
    split,
    condition,
):
    ctx = callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    global model, L, V, df, new_ngram
    global analysis_mode, current_model, current_tokens, current_windows, python_metrics, current_L, current_w_s_val

    # ------------------------------
    # HANDLE "Analyze code" BUTTON
    # ------------------------------
    if triggered_id == "analyze_code":
        code = uploaded_files.get(filename, "")
        # split into tokens & comments
        tokens = tokenize_mixed_content(code, filename)

        # build ngrams if n_size > 1
        n = int(n_size or 1)
        if n > 1:
            data = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        else:
            data = tokens

        # total length
        L = len(data)

        # window parameters
        if definition == "dynamic":
            w_max_val = max(10, int(L / 10))
            w_s_val = max(1, int(w_max_val / 10))
            w_e_val = w_s_val
        else:
            w_s_val = max(1, int(w_s or 1))
            w_max_val = max(w_s_val + 1, int(w_max or L))
            w_e_val = max(1, int(w_e or w_s_val))

        windows = list(range(w_s_val, w_max_val, w_e_val)) or [w_s_val]

        # build frequency model
        local_model: Dict[Any, Ngram] = {}
        for idx, gram in enumerate(data):
            if gram not in local_model:
                ng = Ngram()
                ng.pos = []
                ng.bool = np.zeros(L, dtype=np.uint8)
                local_model[gram] = ng
            local_model[gram].pos.append(idx)
            local_model[gram].bool[idx] = 1

        # apply f_min filter
        fmin_val = int(f_min or 1)
        valid_keys = [tok for tok in local_model if len(local_model[tok].pos) >= fmin_val]

        # compute metrics
        pm = {}
        start = time()
        for tok in valid_keys:
            ng = local_model[tok]
            dt = calculate_distance(
                np.array(ng.pos, dtype=np.uint32),
                L, condition, tok, int(min_dist_option)
            )

            fa_vals = []
            for w in windows:
                counts = make_windows(
                    ng.bool, wi=w, l=L, wsh=w_s_val,
                    overlap_mode=overlap_mode,
                    min_window=(w_s_val if overlap_mode != "overlapping" else None),
                    window_expansion=(w_e_val if overlap_mode != "overlapping" else None)
                )
                fa_vals.append(mse(counts))

            try:
                c, _ = curve_fit(fit, windows, fa_vals, method="lm", maxfev=5000)
                a_val = round(c[0], 8)
                gamma_val = round(c[1], 8)
                fit_vals = [fit(w, *c) for w in windows]
                goodness = round(r2_score(fa_vals, fit_vals), 5)
            except:
                a_val = gamma_val = goodness = 0.0
                fit_vals = [0] * len(windows)

            R_val = round(R(dt), 8)

            pm[tok] = {
                "dt": dt,
                "fa_vals": fa_vals,
                "fit_vals": fit_vals,
                "R": R_val,
                "a": a_val,
                "gamma": gamma_val,
                "goodness": goodness
            }

        execution_time = time() - start

        # build datatable records
        records = []
        for rank, tok in enumerate(valid_keys, start=1):
            m = pm[tok]
            records.append({
                "rank": rank,
                "ngram": tok,
                "F": len(local_model[tok].pos),
                "R": m["R"],
                "a": m["a"],
                "gamma": m["gamma"],
                "goodness": m["goodness"]
            })

        # stash globals for the plots
        analysis_mode = "code"
        current_model = local_model
        current_tokens = valid_keys
        current_windows = windows
        python_metrics = pm
        current_L = L
        current_w_s_val = w_s_val
        V = len(valid_keys)

        # make sure we can save code results later
        import pandas as _pd
        global df
        df = _pd.DataFrame(records)

        return (
            records,
            dash.no_update,  # keep existing chain plot
            {"display": "inline"},  # show table
            {"display": "none"},  # hide chain tab
            dash.no_update,  # no alert
            f"Vocabulary: {V}",
            f"Time: {execution_time:.4f} s",
            False  # close any open toast
        )

    # ------------------------------
    # HANDLE "Analyze natural text" BUTTON
    # (chain_button logic remains unchanged)
    # ------------------------------
    elif triggered_id == "chain_button":
        analysis_mode = "text"
        python_metrics.clear()

        # clear caches & memory
        if hasattr(prepare_data, "clear_cache"):
            prepare_data.clear_cache()
        if hasattr(make_markov_chain, "clear_cache"):
            make_markov_chain.clear_cache()
        clear_memory(keep=["data", "uploaded_files", "file_lengths"])

        # guard
        if chain_clicks is None or dataframe is None:
            return (
                dash.no_update,
                dash.no_update,
                {"display": "none"},
                {"display": "none"},
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        raw = uploaded_files.get(filename, "")
        data = prepare_data(raw, n_size, split)
        L = len(data)

        # — dynamic text (unchanged) —
        if definition == "dynamic":
            # … your existing dynamic‐text logic …
            return (
                df_table,
                dash.no_update,
                {"display": "inline"},
                {"display": "none"},
                dash.no_update,
                vocab_info,
                time_info,
                False,
            )

        # — static (Markov‐style) text —————————————
        start = time()

        # 1) build the markov‐chain model
        model = make_markov_chain(data, order=n_size)

        # 2) build the initial DataFrame of tokens ≥ f_min
        df_local = make_dataframe(model, f_min)

        # 3) window settings (we don’t actually use them in the static scatter,
        #    but we store them so the ∆F vs w plot still works if you switch tabs)
        w_s_val = max(1, int(w_s or 1))
        w_max_val = max(w_s_val + 1, int(w_max or L))
        w_e_val = max(1, int(w_e or w_s_val))
        windows = list(range(w_s_val, w_max_val, w_e_val)) or [w_s_val]

        # 4) compute distance & fluctuation for each token safely
        results = []
        for tok in df_local["ngram"]:
            try:
                ng = model[tok]
                # distances
                dt = calculate_distance(
                    np.array(ng.pos, dtype=np.uint32),
                    L,
                    condition,
                    tok,
                    int(min_dist_option),
                )
                # build ∆F vs w
                fa_vals = []
                fit_vals = []
                for w in windows:
                    cnts = make_windows(
                        ng.bool,
                        wi=w,
                        l=L,
                        wsh=w_s_val,
                        overlap_mode=overlap_mode,
                        min_window=(w_s_val if overlap_mode != "overlapping" else None),
                        window_expansion=(w_e_val if overlap_mode != "overlapping" else None),
                    )
                    fa_vals.append(mse(cnts))
                # curve fit
                try:
                    c, _ = curve_fit(fit, windows, fa_vals, method="lm", maxfev=5000)
                    a_val = round(c[0], 8)
                    gamma_val = round(c[1], 8)
                    fv = [fit(w, *c) for w in windows]
                    goodness = round(r2_score(fa_vals, fv), 5)
                except:
                    a_val = gamma_val = goodness = 0.0
                    fv = []
                R_val = round(R(dt), 8)

            except KeyError:
                # skip any token not actually in model
                continue

            # stash metrics for plotting
            python_metrics[tok] = {
                "fa_vals": fa_vals,
                "fit_vals": fv,
                "R": R_val,
                "gamma": gamma_val,
            }

            # record for DataTable
            results.append(
                {
                    "ngram": tok,
                    "F": len(ng.pos),
                    "R": R_val,
                    "a": a_val,
                    "gamma": gamma_val,
                    "goodness": goodness,
                }
            )

        # 5) build the final records list & globals
        records = []
        for i, rec in enumerate(results, start=1):
            rec["rank"] = i
            records.append(rec)

        current_model = model
        current_windows = windows
        current_L = L
        V = len(results)
        df = pd.DataFrame(records)

        execution_time = time() - start

        return (
            records,
            dash.no_update,
            {"display": "inline"},
            {"display": "none"},
            dash.no_update,
            f"Vocabulary: {V}",
            f"Time: {execution_time:.4f} s",
            False,
        )

    else:
        raise exceptions.PreventUpdate


clikced_ngram = None


from dash import exceptions
import numpy as np
import plotly.graph_objs as go

PAGE_SIZE = 50  # adjust if your table.page_size ever changes

from dash import exceptions
import numpy as np
import plotly.graph_objs as go

PAGE_SIZE = 50  # match your DataTable page_size

from dash import exceptions
import numpy as np
import plotly.graph_objs as go

PAGE_SIZE = 50  # match your DataTable page_size

@app.callback(
    [Output("graphs", "figure"),
     Output("fa",     "figure")],
    [Input("dataframe",            "active_tab"),
     Input("card-tabs",            "active_tab"),
     Input("table",                "active_cell"),
     Input("table",                "page_current"),
     Input("table",                "derived_virtual_data"),
     Input("table",                "derived_virtual_indices"),
     Input("chain",                "clickData"),
     Input("scale",                "value"),
     Input("fa",                   "clickData"),
     Input("graphs",               "clickData"),
     Input("w_max",                "value")],
    [State("n_size", "value"),
     State("def", "value"),
     State("file-selector", "value")]
)
def tab_content(active_tab2, active_tab1, active_cell, page_current,
                derived_virtual_data, derived_virtual_indices,
                click_chain, scale, fa_click, click_dist, w_max,
                n, definition, selected_file):

    global df, current_model, current_windows, python_metrics, current_L

    # nothing to do until df exists
    if df is None or df.empty:
        raise exceptions.PreventUpdate

    # — determine which token was clicked —
    ddata    = derived_virtual_data   or []
    dindices = derived_virtual_indices or []

    if active_cell:
        page   = page_current or 0
        row    = active_cell["row"] or 0
        offset = page * PAGE_SIZE + row

        if offset < len(ddata):
            tok = ddata[offset].get("ngram")
        elif offset < len(dindices):
            orig = dindices[offset]
            tok  = df["ngram"].iat[orig]
        else:
            tok = df["ngram"].iat[0]
    else:
        tok = df["ngram"].iat[0]

    # — DISTRIBUTION PLOT —
    x = np.arange(current_L)
    x_min, x_max = x.min(), x.max()
    span = x_max - x_min or 1

    # pick a fraction of the span (0.005 == 0.5%)
    fraction = 0.005
    bar_width = span * fraction

    fig_dist = go.Figure()
    if tok in current_model:
        fig_dist.add_trace(go.Bar(
            x=x,
            y=current_model[tok].bool,
            name=str(tok),
            width=bar_width,  # ← dynamic
            marker={"line": {"width": 1.5}}
        ))

    fig_dist.update_layout(
        title=f"Positions of “{tok}”",
        bargap=0
    )

    # — ∆F vs w   OR   γ vs R —
    fig_fa = go.Figure()

    if active_tab1 == "tab2":
        # fluctuation‐versus‐window
        fa_vals  = python_metrics.get(tok, {}).get("fa_vals", [])
        fit_vals = python_metrics.get(tok, {}).get("fit_vals", [])
        fig_fa.add_trace(go.Scatter(
            x=current_windows, y=fa_vals,
            mode="markers", name="∆F"
        ))
        if fit_vals:
            fig_fa.add_trace(go.Scatter(
                x=current_windows, y=fit_vals,
                name="fit=aw^b"
            ))
        title = f"∆F vs w for “{tok}”"

    else:
        # build full cloud of (R,γ) with hover labels
        xs, ys, labels = [], [], []
        for t, m in python_metrics.items():
            Rv = m.get("R", None)
            Gv = m.get("gamma", None)
            if Rv is not None and Gv is not None:
                xs.append(Rv)
                ys.append(Gv)
                labels.append(t)

        fig_fa.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name="all tokens",
            marker={"opacity": 0.5, "size": 8},
            text=labels,
            hovertemplate="%{text}<br>R: %{x:.3f}<br>γ: %{y:.3f}<extra></extra>"
        ))

        # overlay selected in red
        R_sel = python_metrics.get(tok, {}).get("R", None)
        G_sel = python_metrics.get(tok, {}).get("gamma", None)
        if R_sel is not None and G_sel is not None:
            fig_fa.add_trace(go.Scatter(
                x=[R_sel], y=[G_sel],
                mode="markers",
                name=str(tok),
                marker={"color": "red", "size": 12, "line": {"width": 2, "color": "darkred"}},
                text=[tok],
                hovertemplate="%{text}<br>R: %{x:.3f}<br>γ: %{y:.3f}<extra></extra>"
            ))

        title = f"γ vs R for “{tok}”"

    fig_fa.update_layout(
        title=title,
        hovermode="closest"
    )
    fig_fa.update_xaxes(type=scale)
    fig_fa.update_yaxes(type=scale)
    if selected_file:
        save_token_to_excel(selected_file, tok)

    return fig_dist, fig_fa



@app.callback(
    [Output("temp_seve", "children")],
    [Input("save", "n_clicks"),
     Input("table", "active_cell"),
     Input("table", "page_current"),
     Input("table", "derived_virtual_indices")],
    [State("file-selector", "value"),
     State("n_size", "value"),
     State("w_min", "value"),
     State("w_s", "value"),
     State("w_e", "value"),
     State("w_max", "value"),
     State("f_min", "value"),
     State("condition", "value"),
     State("def", "value"),
     State("min_dist_option", "value"),
     State("overlap_mode", "value")]
)
def save(n_clicks, active_cell, page_current, ids,
         filename, n_size, w_min, w_s, w_e, w_max,
         fmin, condition, definition, min_dist_option, overlap_mode):
    ctx = callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate
    # only proceed when the save‐button caused this
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger != "save":
        raise exceptions.PreventUpdate

    if n_clicks is None:
        return dash.no_update
    if not filename:
        return [html.Div(["No file selected to save"])]

    try:
        save_folder = "saved_data"
        os.makedirs(save_folder, exist_ok=True)

        global df, model, new_ngram
        messages = []

        if definition == "dynamic":
            # Save the dynamic‐mode table (including new_ngram row)
            df_to_save = df.copy()
            base = (f"{filename} condition={condition},fmin={fmin},n={n_size},"
                    f"w=({w_min},{w_s},{w_e},{w_max}),definition={definition},"
                    f"min_dist={min_dist_option},overlap={overlap_mode}")
            path_main = os.path.join(save_folder, f"{base}.xlsx")
            unique_main = get_unique_path(path_main)
            df_to_save.to_excel(unique_main, index=False)
            messages.append(f"Saved main data to {unique_main}")

            # also save new_ngram details if present
            if new_ngram and hasattr(new_ngram, "dfa"):
                df_details = pd.DataFrame({
                    "w": list(new_ngram.dfa.keys()),
                    "∆F": list(new_ngram.dfa.values()),
                    "fit=a*w^b": new_ngram.temp_dfa
                })
                details_base = f"{filename}_new_ngram_details"
                path_det = os.path.join(save_folder, f"{details_base}.xlsx")
                unique_det = get_unique_path(path_det)
                df_details.to_excel(unique_det, index=False)
                messages.append(f"Saved new_ngram details to {unique_det}")

        else:
            # Non-dynamic mode: compute the eight summary metrics
            df_copy = df[df.ngram != "new_ngram"].copy()
            df_copy['w'] = df_copy['F'] / df_copy['F'].sum()

            R_avg    = df_copy['R'].mean()
            dR       = df_copy['R'].std()
            Rw_avg   = (df_copy['R'] * df_copy['w']).sum()
            dRw      = np.sqrt(((df_copy['R'] - Rw_avg)**2 * df_copy['w']).sum())
            gamma_avg= df_copy['gamma'].mean()
            dgamma   = df_copy['gamma'].std()
            gammaw_avg = (df_copy['gamma'] * df_copy['w']).sum()
            dgammaw  = np.sqrt(((df_copy['gamma']-gammaw_avg)**2 * df_copy['w']).sum())

            # stick them into the first row
            for col, val in [
                ("R_avg", R_avg), ("dR", dR),
                ("Rw_avg", Rw_avg), ("dRw", dRw),
                ("gamma_avg", gamma_avg), ("dgamma", dgamma),
                ("gammaw_avg", gammaw_avg), ("dgammaw", dgammaw)
            ]:
                df_copy[col] = None
                df_copy.loc[df_copy.index[0], col] = val

            df_copy = df_copy.drop(columns=["w"])
            base = (f"{filename} condition={condition},fmin={fmin},n={n_size},"
                    f"w=({w_min},{w_s},{w_e},{w_max}),definition={definition},"
                    f"min_dist={min_dist_option},overlap={overlap_mode}")
            path_main = os.path.join(save_folder, f"{base}.xlsx")
            unique_main = get_unique_path(path_main)
            df_copy.to_excel(unique_main, index=False)
            messages.append(f"Saved!")

        return [html.Div([html.Div(msg) for msg in messages])]
    except Exception as e:
        return [html.Div([f"Error saving data: {e}"])]


def pick_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_selected = filedialog.askdirectory()
    root.destroy()

    if folder_selected:
        return folder_selected
    else:
        return None


# import webbrowser # Commented out as it might cause issues if run non-interactively

if __name__ == "__main__":
    webbrowser.open_new("http://127.0.0.1:8050/")  # Автоматично відкриває браузер
    # Replace app.run() with the older style Flask server run for Dash < 2.0
    app.server.run(host='0.0.0.0', port=8050, debug=False)


# Add callback to toggle batch window settings
@app.callback(
    Output("batch_custom_controls", "is_open"),
    [Input("batch_window_mode", "value")]
)
def toggle_batch_window_controls(mode):
    return mode in ["ui", "auto"]
