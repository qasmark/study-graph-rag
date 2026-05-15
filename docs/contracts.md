# Как работать с Eval API (для студентов)

Сервис проверяет ваш Graph RAG: вы забираете вопросы, прогоняете систему по статьям и отправляете найденные цитаты. Сервер сравнивает их с эталоном и возвращает метрики.

Адрес сервера и ключ выдаёт преподаватель. Полная схема полей - в Swagger: `http://<host>:8000/docs`.

---

## Что вы делаете по шагам

1. Получаете персональный API-ключ (показывают **один раз** - сохраните).
2. Запрашиваете список вопросов.
3. Локально прогоняете Graph RAG.
4. Отправляете цитаты на оценку.
5. Смотрите метрики в ответе (при необходимости - повторно по `submission_id`).

**Важно:** в текущей версии оцениваются только **цитаты** (упорядоченный список фрагментов из статьи). Поле `answer` можно приложить, но на метрики оно **не влияет**.

---

## Авторизация

Во всех студенческих запросах нужен заголовок:

```
X-API-Key: <ваш_ключ>
```

Для POST с телом добавьте:

```
Content-Type: application/json
```

Админский заголовок `X-Admin-Key` студентам не нужен.

---

## Ошибки сервера

| Код | Когда возникает | Что делать |
|-----|-----------------|------------|
| **401** | Нет заголовка `X-API-Key`, ключ неверный или отозван | Проверьте ключ у преподавателя, нет ли лишних пробелов в заголовке |
| **404** | Запрошен `submission_id`, которого нет, или он создан другим ключом | Используйте `submission_id` из своего ответа на submit и тот же API-ключ |
| **422** | Тело запроса не JSON или не совпадает со схемой (нет `items`, неверные типы полей) | Сверьтесь с примерами ниже и с `/docs` |
| **429** | Больше **10 запросов в минуту** на один ключ | Подождите; в ответе будет заголовок `Retry-After` (секунды) |
| **503** | Слишком много запросов на весь сервер сразу (защита от burst) | Подождите ~минуту и не шлите десятки запросов в секунду |

Типичное тело ошибки FastAPI:

```json
{
  "detail": "Missing or invalid X-API-Key"
}
```

Для rate limit:

```json
{
  "detail": "Rate limit exceeded (10 requests per minute)"
}
```

**Практика:** один полный прогон = один раз вопросы + один раз submit. При отладке метрики считайте у себя, а не долбите API в цикле.

---

## 1. Получить вопросы

**Запрос:**

```http
GET /v1/eval/questions
X-API-Key: YOUR_STUDENT_KEY
```

**Пример curl:**

```bash
curl -s http://localhost:8000/v1/eval/questions \
  -H "X-API-Key: YOUR_STUDENT_KEY"
```

**Пример ответа** (эталонных ответов и цитат здесь нет - намеренно):

```json
{
  "version": "1.0",
  "count": 24,
  "items": [
    {
      "id": "vitro_q01",
      "question": "Какие существуют способы использования LLM для прогноза и решения задач временных рядов (TS), о которых идет речь в статье «VITRO: Vocabulary Inversion for Time-series Representation Optimization»?",
      "article_id": "vitro",
      "article_title": "VITRO: Vocabulary Inversion for Time-series Representation Optimization"
    }
  ]
}
```

**Поля в каждом элементе `items`:**

- `id` - идентификатор вопроса; при отправке ответа указываете его как `question_id`.
- `question` - текст вопроса с привязкой к одной статье.
- `article_id` - короткий код статьи (`vitro`, `time-llm`, …).
- `article_title` - полное название; должно совпадать с `Article.title` в вашем графе.

Сейчас в наборе **24 вопроса** (12 статей × 2 вопроса). Версия набора - в поле `version`.

---

## 2. Отправить ответы

**Запрос:**

```http
POST /v1/eval/submit
X-API-Key: YOUR_STUDENT_KEY
Content-Type: application/json
```

**Тело (что вы шлёте):**

```json
{
  "run_id": "lab3-attempt-2",
  "items": [
    {
      "question_id": "vitro_q01",
      "answer": "Две парадигмы: LLM-for-TS и TS-for-LLM...",
      "citations": [
        "This work summarizes two ways to accomplish Time-Series (TS) tasks in today's Large Language Model (LLM) context:",
        "TS-for-LLM (data-centric, modify TS). Based on the existing LLMs, furthest freezing them..."
      ]
    },
    {
      "question_id": "vitro_q02",
      "citations": [
        "We present accuracy scores for all 128 kinds of univariate TS datasets in UCR archive"
      ]
    }
  ]
}
```

| Поле | Обязательно? | Смысл |
|------|--------------|--------|
| `run_id` | Нет | Ваша метка прогона (лаба, дата) |
| `items` | **Да** | Массив ответов по вопросам |
| `items[].question_id` | **Да** | Тот же `id`, что в `/questions` |
| `items[].citations` | **Да** (может быть `[]`) | Упорядоченный список цитат; **порядок = ранг** |
| `items[].answer` | Нет | Текст ответа; **не оценивается** |

**Пример curl (один вопрос):**

```bash
curl -s -X POST http://localhost:8000/v1/eval/submit \
  -H "X-API-Key: YOUR_STUDENT_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\":\"demo-1\",\"items\":[{\"question_id\":\"vitro_q01\",\"citations\":[\"This work summarizes two ways\"]}]}"
```

**Пример на Python (httpx):**

```python
import httpx

BASE = "http://localhost:8000"
headers = {"X-API-Key": "YOUR_STUDENT_KEY"}

questions = httpx.get(f"{BASE}/v1/eval/questions", headers=headers).json()

items = []
for q in questions["items"]:
    # TODO: ваш Graph RAG → список citations для q["id"]
    items.append({"question_id": q["id"], "citations": [...]})

submit = httpx.post(
    f"{BASE}/v1/eval/submit",
    headers=headers,
    json={"run_id": "my-run-1", "items": items},
)
print(submit.json()["report"]["aggregate"])
```

Можно отправить не все 24 вопроса, а часть - метрики в `aggregate` усреднятся только по присланным. Для зачёта обычно нужен полный набор. Неизвестные `question_id` не ломают запрос - они попадут в `unknown_question_ids` в отчёте.

---

## Требования к цитатам

Цитата - **фрагмент из текста статьи**, а не пересказ. Обычно 1–3 предложения из PDF; часто на английском.

- Все цитаты по одному вопросу - **только из одной статьи** (`article_title` этого вопроса).
- **Порядок важен:** первая цитата - топ-1, метрики смотрят на первые 5 и 10 позиций.
- Имеет смысл отдавать до **10** цитат; ниже 10-й позиции ранг не учитывается.
- Пустой `citations: []` допустим, но метрики будут нулевые.

---

## Graph RAG (что ожидается от системы)

Сервер не проверяет код графа, только цитаты. Но задание предполагает:

- сущность **Article** с полем **title** = `article_title` из API;
- по вопросу - найти нужную статью и искать цитаты **только в ней**, не по всему корпусу;
- на выход - упорядоченный список пассажей → поле `citations`.

Типичные ошибки: поиск по всем статьям сразу; укороченное название в графе; summary вместо дословной цитаты из PDF.

---

## 3. Ответ после submit (метрики)

**Пример ответа:**

```json
{
  "submission_id": "550e8400-e29b-41d4-a716-446655440000",
  "run_id": "lab3-attempt-2",
  "report": {
    "aggregate": {
      "recall_at_5": 0.72,
      "recall_at_10": 0.85,
      "precision_at_5": 0.64,
      "precision_at_10": 0.58,
      "f1_at_5": 0.68,
      "f1_at_10": 0.69,
      "ndcg_at_5": 0.75,
      "ndcg_at_10": 0.81
    },
    "citation_validity_rate": 0.95,
    "mean_gold_covered_in_top_10": 2.1,
    "per_question": [
      {
        "question_id": "vitro_q01",
        "article_id": "vitro",
        "metrics": { "recall_at_10": 0.85, "ndcg_at_10": 0.81 },
        "gold_citation_count": 3,
        "predicted_count": 2,
        "gold_covered_in_top_10": 2,
        "citation_validity_rate": 1.0,
        "invalid_citations": [],
        "predicted_details": [
          {
            "citation": "This work summarizes two ways...",
            "is_relevant": true,
            "is_valid": true,
            "fuzzy_score": 98.5,
            "matched_gold_index": 0
          }
        ]
      }
    ],
    "unknown_question_ids": []
  }
}
```

**Кратко по метрикам** (числа от 0 до 1, выше - лучше):

- **Recall@5 / @10** - какую долю эталонных цитат вы нашли в первых 5 / 10 позициях вашего списка.
- **Precision@5 / @10** - какая доля ваших первых 5 / 10 цитат оказалась удачной.
- **F1@5 / @10** - баланс precision и recall.
- **nDCG@5 / @10** - учитывает ещё и **порядок** (релевантные выше - лучше).
- **citation_validity_rate** - доля цитат, которые реально похожи на текст статьи.
- **invalid_citations** - цитаты, не прошедшие проверку «из текста статьи».

Релевантность: совпадение с эталоном после нормализации текста или fuzzy ≥ 90%. Валидность: фрагмент есть в тексте статьи или в корпусе цитат по этой статье.

---

## 4. Получить отчёт повторно

```http
GET /v1/eval/submissions/{submission_id}
X-API-Key: YOUR_STUDENT_KEY
```

```bash
curl -s http://localhost:8000/v1/eval/submissions/550e8400-e29b-41d4-a716-446655440000 \
  -H "X-API-Key: YOUR_STUDENT_KEY"
```

Формат ответа такой же, как у `POST /v1/eval/submit`. Чужой `submission_id` или другой ключ → **404**.

---

## Чеклист перед сдачей

- [ ] `Article.title` в графе = `article_title` из API
- [ ] Цитаты из одной статьи на вопрос, порядок от лучшей к худшей
- [ ] Все 24 `question_id` без опечаток
- [ ] Ключ не в git и не в публичном чате
- [ ] Не больше 10 запросов в минуту к API

---

## Чего в этой версии нет

- Оценка поля `answer` (только цитаты).
- Вопросы с цитатами из нескольких статей в одном ответе.
- Парафразы вопросов в тестовом наборе.
- Асинхронная очередь - ответ на submit приходит сразу.

При обновлении набора изменится `version` в `/questions` - смотрите объявления преподавателя.
