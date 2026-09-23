# Arquitetura RAG (Retrieval-Augmented Generation)
Este documento apresenta uma visão técnica e prática sobre a implementação de sistemas de Geração Aumentada por Recuperação (RAG).

---

## 1. O que é RAG?
RAG (Retrieval-Augmented Generation) é uma técnica de inteligência artificial que conecta modelos de linguagem grande (LLMs) a bases de dados externas antes de responder a uma pergunta. Isso reduz alucinações e garante respostas baseadas em dados atualizados ou privados.

---

## 2. A Arquitetura em 3 Pilares Técnicos

### Passo 1: O Pipeline de Ingestão (Data Ingestion & Embedding)
Antes de qualquer busca acontecer, você precisa preparar os dados brutos (PDFs, Markdown, logs).
- **Processo:** O texto é dividido em blocos menores com sobreposição (chunking) e enviado para uma API de embedding que transforma o texto em um array numérico (vetor) que representa seu significado semântico. Os vetores são salvos em um banco de dados especializado.

### Passo 2: A Busca Semântica e Reranking (Retrieval)
A busca não utiliza correspondência de palavras-chave exatas, mas sim a similaridade matemática entre os vetores.
- **Processo:** A pergunta do usuário é convertida em vetor. O banco vetorial calcula a similaridade (como a distância de cosseno) e retorna os blocos mais próximos. Ferramentas adicionais de Reranking podem reordenar os resultados para maior precisão.

### Passo 3: A Janela de Contexto Dinâmica (Generation)
A IA atua apenas como um motor de raciocínio.
- **Processo:** O código reconstrói o prompt e "injeta" os blocos recuperados diretamente na mensagem do sistema enviado à LLM, forçando o modelo a responder estritamente com base nas informações fornecidas.

---

## 3. Implementação Prática em Python

```python
import os
import numpy as np
from openai import OpenAI

# Inicializa o cliente da OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sua-chave-aqui"))

# PASSO 1: Ingestão de Dados e Geração de Embeddings
DOCUMENTACAO_CHUNKS = [
    "Para instalar o SDK de Pagamentos, execute o comando: npm install @pay-sdk/core.",
    "O erro 403 (Forbidden) no SDK de Pagamentos ocorre quando as chaves de API estão inválidas ou expiradas.",
    "O erro 500 indica instabilidade nos servidores internos de pagamento.",
    "Para autenticar as requisições, configure o cliente usando: PayClient(api_key='sua_chave')."
]

def obter_embedding(texto: str, modelo: str = "text-embedding-3-small") -> list:
    resposta = client.embeddings.create(input=[texto], model=modelo)
    return resposta.data.embedding

print("⚙️ Gerando base de conhecimento vetorial...")
banco_vetorial = {
    chunk: np.array(obter_embedding(chunk)) for chunk in DOCUMENTACAO_CHUNKS
}

# PASSO 2: Busca Semântica (Retrieval)
def calcular_similaridade_cosseno(vetor_a, vetor_b) -> float:
    return np.dot(vetor_a, vetor_b) / (np.linalg.norm(vetor_a) * np.linalg.norm(vetor_b))

def buscar_contexto_relevante(pergunta_usuario: str, limite_resultados: int = 2) -> list:
    vetor_pergunta = np.array(obter_embedding(pergunta_usuario))
    scores = []
    for chunk, vetor_chunk in banco_vetorial.items():
        similaridade = calcular_similaridade_cosseno(vetor_pergunta, vetor_chunk)
        scores.append((chunk, similaridade))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scores[:limite_resultados]]

pergunta = "Estou recebendo erro 403 ao tentar rodar a API de pagamento. O que pode ser?"
contextos_encontrados = buscar_contexto_relevante(pergunta, limite_resultados=2)

# PASSO 3: Geração com Contexto Injetado (Generation)
contexto_consolidado = "\n".join([f"- {c}" for c in contextos_encontrados])

resposta_llm = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "Responda à pergunta baseando-se estritamente no contexto fornecido."
        },
        {
            "role": "user",
            "content": f"CONTEXTO:\n{contexto_consolidado}\n\nPERGUNTA:\n{pergunta}"
        }
    ],
    temperature=0.2
)

print("\n=== RESPOSTA FINAL ===")
print(resposta_llm.choices.message.content)
```

---

## 4. Como Obter a Chave da API OpenAI

1. Acesse o portal [OpenAI API](https://platform.openai.com/).
2. Faça login ou crie sua conta de desenvolvedor.
3. No menu lateral, acesse **API Keys**.
4. Clique em **+ Create new secret key**, dê um nome e salve a chave em local seguro.
5. Configure como variável de ambiente no seu terminal:
   - **Linux/macOS:** `export OPENAI_API_KEY="sua_chave"`
   - **Windows:** `set OPENAI_API_KEY="sua_chave"`
