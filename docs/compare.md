# Базовая модель против адаптера

База: `Qwen/Qwen3-0.6B`. Адаптер: `models/adapter_all_layers`.
Генерация жадная, до 40 токенов; режим рассуждений: `false`.

Пять примеров показывают изменение ответов после подключения адаптера. Они не являются статистической оценкой качества классификации.

## Системная инструкция

Определи жанры фильма по его описанию. Ответь только списком жанров через запятую.

## 1. Описание фильма

Francis, a young man, recalls in his memory the horrible experiences he and his fiancée Jane recently went through. Francis and his friend Alan visit The Cabinet of Dr. Caligari, an exhibit where the mysterious doctor shows the somnambulist Cesare, and awakens him for some moments from his death-like sleep.

Источник: `movie_de2347cf6a48feb5f30d587fbf181485daa20be1b1394cb64020c0a2a9e17a92`.

**Эталонная метка датасета:** драма, криминал, триллер, ужасы

**База:**

> - Фильм
> - Драма
> - Сюжет-драм
> - Стиль-драм

**Адаптер:**

> драма, триллер, ужасы

## 2. Описание фильма

During America’s Civil War, Union spies steal engineer Johnnie Gray's beloved locomotive, 'The General'—with Johnnie's lady love aboard an attached boxcar—and he single-handedly must do all in his power to both get The General back and to rescue Annabelle.

Источник: `movie_a120cdc23310ceb38837208e1234d5322e478f08c98a1f9c14a8d1605b86c22f`.

**Эталонная метка датасета:** боевик, военный, драма, комедия, приключения

**База:**

> - Фильм
> - Сюжет
> - Сериал

**Адаптер:**

> боевик, комедия, приключения, семейный

## 3. Описание фильма

In a futuristic city sharply divided between the working class and the city planners, the son of the city's mastermind falls in love with a working class prophet who predicts the coming of a savior to mediate their differences.

Источник: `movie_c88d68979ad500edecb4b92adc824f816a170ad8cab7b4bbaa62d43ae665110c`.

**Эталонная метка датасета:** драма, фантастика

**База:**

> футурный, драматический, сюжетный.

**Адаптер:**

> драма, мелодрама, фантастика

## 4. Описание фильма

Un Chien Andalou is an European avant-garde surrealist film, a collaboration between director Luis Buñuel and Salvador Dali.

Источник: `movie_72f8fdbe7fbe2a2fcfb7239ed242a11018ea8d18625a1e3969ab80f6c5c60cd9`.

**Эталонная метка датасета:** ужасы, фэнтези

**База:**

> - avant-garde
> - surrealist

**Адаптер:**

> документальный

## 5. Описание фильма

Dr. Henry Jekyll believes that there are two distinct sides to men - a good and an evil side. He believes that by separating the two, man can become liberated. He succeeds in his experiments with chemicals to accomplish this and transforms into Hyde to commit horrendous crimes. When he discontinues use of the drug, it is already too late.

Источник: `movie_1d66bcfd17681eacff8d85ac9d2f05ee34fa7512ecb1c02fa428bff79f65f71b`.

**Эталонная метка датасета:** ужасы, фантастика

**База:**

> - фантастический
> - научный
> - психологический

**Адаптер:**

> драма, триллер, ужасы
