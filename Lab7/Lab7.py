import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

# 0. Глобальные параметры
IMG_SIZE = 28
N_CLASSES = 10
N_EPOCHS = 30
BATCH_SIZE = 128
N_IMAGES = 25
DATA_DIR = 'input/'
OUTPUT_DIR = 'output/'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'digit_model.keras')

# Создаем папку для вывода, если ее нет
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Функции загрузки и подготовки данных

def get_data():
    """Загрузка данных MNIST из бинарных файлов"""
    print(f"Загрузка данных из: {DATA_DIR}")

    try:
        x_train = np.fromfile(os.path.join(DATA_DIR, 'images_trn.bin'), dtype=np.uint8)
        y_train = np.fromfile(os.path.join(DATA_DIR, 'labels_trn.bin'), dtype=np.uint8)
        x_test = np.fromfile(os.path.join(DATA_DIR, 'images_tst.bin'), dtype=np.uint8)
        y_test = np.fromfile(os.path.join(DATA_DIR, 'labels_tst.bin'), dtype=np.uint8)
        print("Данные успешно загружены.")

        # Придание правильной формы
        train_size = len(y_train)
        test_size = len(y_test)

        x_train = x_train.reshape(train_size, IMG_SIZE, IMG_SIZE)
        x_test = x_test.reshape(test_size, IMG_SIZE, IMG_SIZE)

    except FileNotFoundError:
        # запасной вариант
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        print("Используется стандартная загрузка Keras/TensorFlow.")

    return x_train, y_train, x_test, y_test


def process_data(x, y):
    """Предобработка данных для классификатора"""
    # Преобразование 28x28 в 784
    x_flat = x.reshape(-1, IMG_SIZE * IMG_SIZE)
    # Нормализация [0, 255] -> [0, 1]
    x_norm = x_flat.astype('float32') / 255
    # One-hot encoding для меток
    y_categorical = tf.keras.utils.to_categorical(y, N_CLASSES)
    return x_norm, y_categorical


def plot_images(x, y, title):
    """Визуализация случайных 25 изображений и сохранение графика"""
    indices = np.random.randint(0, len(x), N_IMAGES)
    x_sample = x[indices]
    y_sample = y[indices]

    plt.figure(figsize=(6, 6))
    for i in range(N_IMAGES):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_sample[i], cmap='gray')  # Убрал .reshape(), т.к. данные уже в форме 28x28
        plt.title(f'{y_sample[i]}', fontsize=9)
        plt.axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    print(f"График '{title}' сохранен: {os.path.join(OUTPUT_DIR, filename)}")
    plt.show()
    plt.close()


# 2. Функции построения и обучения модели

def create_model(layers=[512, 256, 128]):
    """Создание модели НС с не менее чем 3 Dense-слоями"""
    input_layer = Input(shape=(IMG_SIZE * IMG_SIZE,))
    x = Dropout(0.3)(input_layer)

    # Не менее 3-х Dense-слоев
    for units in layers:
        x = Dense(units, activation='relu')(x)
        x = Dropout(0.5)(x)

    output_layer = Dense(N_CLASSES, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_training(history):
    """Вывод истории обучения и сохранение графика"""
    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Потери (Обучение)')
    plt.plot(history['val_loss'], label='Потери (Валидация)')
    plt.xlabel('Эпоха');
    plt.ylabel('Потери');
    plt.title('График потерь')
    plt.legend();
    plt.grid(True)

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Точность (Обучение)')
    plt.plot(history['val_accuracy'], label='Точность (Валидация)')
    plt.xlabel('Эпоха');
    plt.ylabel('Точность');
    plt.title('График точности')
    plt.legend();
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=150)
    print(f"График истории обучения сохранен: {plot_path}")
    plt.show()
    plt.close()


def compare_models(x_train, y_train, x_test, y_test):
    """Исследование зависимости точности от числа параметров"""
    configs = [
        [32],  # 1 слой
        [128, 64],  # 2 слоя
        [512, 256, 128],  # 3 слоя
        [1024, 512, 256, 128]  # 4 слоя
    ]
    names = [
        "Маленькая [32]",
        "Средняя [128, 64]",
        "Основная [512, 256, 128]",
        "Очень большая [1024, 512, 256, 128]"
    ]

    results = []
    print("\nИсследование зависимости точности от числа параметров (10 эпох):")

    for config, name in zip(configs, names):
        model = create_model(config)
        n_params = model.count_params()

        # Обучение только на 10 эпохах для сравнения
        h = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=10,
                      validation_data=(x_test, y_test), verbose=0)

        val_acc = h.history['val_accuracy'][-1]
        results.append({'name': name, 'params': n_params, 'val_accuracy': val_acc})
        print(f"  Модель '{name}' (Параметры: {n_params:,}) - Точность: {val_acc:.4f}")

    # Вывод таблицы
    print("\nТАБЛИЦА: ЗАВИСИМОСТЬ ТОЧНОСТИ ОТ ЧИСЛА ПАРАМЕТРОВ")
    print("=" * 70)
    print(f"{'Модель':<35} {'Параметры':<15} {'Точность (тест)':<15}")
    print("-" * 70)
    for res in sorted(results, key=lambda x: x['params']):
        print(f"{res['name']:<35} {res['params']:<15,} {res['val_accuracy']:<15.4f}")
    print("=" * 70)


# Функции визуализации предсказаний

def show_predictions(model, x_sample, y_true, save_dir, epoch, n_images=N_IMAGES):
    """Функция для визуализации предсказаний"""
    if epoch == 1 or epoch % 10 == 0:

        print(f"\n[VIZ] Генерация предсказаний для Эпохи {epoch}...")

        # Предсказание на фиксированных примерах
        y_pred_proba = model.predict(x_sample, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Визуализация
        cols = 5
        rows = int(np.ceil(n_images / cols))

        plt.figure(figsize=(10, rows * 2))

        for i in range(n_images):
            plt.subplot(rows, cols, i + 1)

            # Изображение для показа (денормализация)
            img = (x_sample[i].reshape(IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8)
            plt.imshow(img, cmap='gray')

            true_label = y_true[i]
            pred_label = y_pred[i]

            # Цветовая индикация
            color = 'green' if true_label == pred_label else 'red'

            plt.title(f'T: {true_label} | P: {pred_label}',
                      color=color, fontsize=10)
            plt.axis('off')

        title = f'Предсказания на тестовом множестве (Эпоха {epoch})'
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        filename = f'predictions_epoch_{epoch:02d}.png'
        plot_path = os.path.join(save_dir, filename)

        # Сохранение и вывод
        plt.savefig(plot_path, dpi=150)
        print(f"[VIZ] График предсказаний сохранен: {plot_path}")
        plt.show()
        plt.close()


def train_with_viz(model, x_train, y_train, x_test, y_test, x_test_raw, epochs):
    """Основная функция обучения с ручным циклом для визуализации предсказаний"""

    # Инициализация истории обучения
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    # Выбор фиксированных примеров для визуализации
    np.random.seed(42)
    viz_indices = np.random.choice(len(x_test_raw), N_IMAGES, replace=False)
    x_viz = x_test[viz_indices]
    y_viz_true = y_test_raw[viz_indices]  # Исправлено: было x_test_raw, должно быть y_test_raw

    # Главный цикл обучения по эпохам
    for epoch in range(1, epochs + 1):
        print(f"\nЭпоха {epoch}/{epochs}")

        # Обучение на 1 эпохе
        h_epoch = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, verbose=2)

        # Оценка на валидационном множестве
        val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)

        # Обновление истории
        history['loss'].extend(h_epoch.history['loss'])
        history['accuracy'].extend(h_epoch.history['accuracy'])
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Вывод результатов текущей эпохи
        print(
            f" - loss: {h_epoch.history['loss'][0]:.4f} - acc: {h_epoch.history['accuracy'][0]:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        # Визуализация
        show_predictions(model, x_viz, y_viz_true, OUTPUT_DIR, epoch)

    return history


# Функции тестирования и метрик

def calc_metrics(y_true, y_pred):
    """Ручной подсчет precision, recall и F1"""
    print("\n--- Ручной расчет метрик")

    metrics = {}
    total_tp = 0
    n_samples = len(y_true)

    print(f"{'Класс':<7} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 47)

    for cls in range(N_CLASSES):
        # Логика для бинарной классификации относительно класса
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)
        total_tp += tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support}
        print(f"{cls:<7} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")

    print("-" * 47)

    # Расчет средних значений
    accuracy = total_tp / n_samples
    macro_f1 = np.mean([m['f1'] for m in metrics.values()])
    weighted_f1 = np.sum([m['f1'] * m['support'] for m in metrics.values()]) / n_samples

    print(f"{'accuracy':<7} {'':<10} {'':<10} {accuracy:<10.4f} {n_samples:<10}")
    print(f"{'macro avg':<7} {'':<10} {'':<10} {macro_f1:<10.4f} {n_samples:<10}")
    print(f"{'weighted avg':<7} {'':<10} {'':<10} {weighted_f1:<10.4f} {n_samples:<10}")


# Главный запуск

# Шаг A: Загрузка и подготовка
x_train_raw, y_train_raw, x_test_raw, y_test_raw = get_data()

# Вывод случайных проверочных рисунков
print("\n--- Визуализация данных")
plot_images(x_train_raw, y_train_raw, title='Обучающее множество (ОМ)')
plot_images(x_test_raw, y_test_raw, title='Проверочное множество (ПМ)')

# Предобработка
x_train_proc, y_train_proc = process_data(x_train_raw, y_train_raw)
x_test_proc, y_test_proc = process_data(x_test_raw, y_test_raw)

# Создание, обучение, сохранение и анализ
print("\n--- Обучение и анализ")

# Создать, обучить, сохранить модель
model = create_model()
model.summary()

# Замена model.fit() на ручной цикл с визуализацией
history_dict = train_with_viz(
    model,
    x_train_proc, y_train_proc,
    x_test_proc, y_test_proc,
    x_test_raw,
    epochs=N_EPOCHS
)

model.save(MODEL_PATH)
print(f"\nМодель сохранена в файле: {MODEL_PATH}")

# Вывести историю обучения
plot_training(history_dict)

# Посмотреть зависимость точности от числа параметров
compare_models(x_train_proc, y_train_proc, x_test_proc, y_test_proc)

# Тестирование и метрики
print("\n--- Тестирование и метрики")

# Загрузить и протестировать обученную модель
try:
    loaded_model = load_model(MODEL_PATH)
    print(f"Модель успешно загружена из {MODEL_PATH}.")

    # Предсказание и преобразование в метки классов (0-9)
    y_pred_proba = loaded_model.predict(x_test_proc)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test_raw

    # Использование classification_report из sklearn
    print("\n--- Отчет о классификации (sklearn classification_report)")
    print(classification_report(y_true, y_pred, digits=4))

    # Написать свою версию вычисления метрик
    calc_metrics(y_true, y_pred)

except Exception as e:
    print(f"Ошибка при загрузке или тестировании модели: {e}")

print("\n--- конец программы ---")