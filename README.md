# Agente Interno de Reembolsos e Cancelamentos (POC RAG)

Assistente de IA para apoiar colaboradores (Foodlovers) em decisões operacionais de **reembolso e
cancelamento**, aplicando estritamente as políticas oficiais. Combina busca semântica (RAG) sobre a base
de políticas, **regras determinísticas em código** e uma LLM restrita a redigir a justificativa dentro de
um schema validado.

> A base `base_conhecimento_ifood_genai-exemplo.csv` é a **base simulada fornecida pelo iFood** para o
> processo seletivo (20 registros, para fins educacionais). Ela **não representa políticas oficiais** do iFood.

## Cenários de teste do README do iFood

| Pergunta | Comportamento do agente |
|---|---|
| "O cliente quer reembolso, mas o pedido já saiu para entrega. Ainda é permitido?" | Sem motivo informado → `AVALIACAO_MANUAL` (R6) com a Política 3.2, indicando como decidir por motivo: falha do restaurante/app (2.1) ou do entregador (2.2) → elegível; desistência (3.2/4.1) → não elegível |
| "O restaurante cancelou por falta de ingrediente. O reembolso é automático?" | Recupera a Política 2.3 (reembolso automático, salvo exceções operacionais) |
| "O cliente foi cobrado após o cancelamento. O que fazer?" | Recupera o Fluxo Financeiro: validar o status do estorno (até 14 dias) e abrir ticket se houver divergência |

## Arquitetura

```
pergunta do colaborador
   │
   ▼
[1] Sanitização + detecção de prompt injection ──(suspeita)──► TRANSBORDO_HUMANO  (sem RAG, sem LLM)
   │
   ▼
[2] RAG: embedding (text-embedding-3-small) + similaridade de cosseno em memória
   │
   ▼
[3] score máximo < THRESHOLD (0.75)? ──(sim)──► Fallback Seguro: TRANSBORDO_HUMANO  (sem LLM)
   │
   ▼
[4] Regras determinísticas (business_rules.py) ──► decisão obrigatória, se aplicável
   │
   ▼
[5] LLM (JSON) com <contexto>, <decisao_regra_negocio> e <mensagem_usuario> isolados
   │
   ▼
[6] Validação Pydantic + enforcement
      • LLM divergiu da regra?           → prevalece a regra
      • fonte citada fora do contexto?   → TRANSBORDO_HUMANO
      • JSON inválido / LLM fora do ar?  → regra determinística, ou TRANSBORDO_HUMANO se não houver regra
   │
   ▼
AgentResponse { decisao, justificativa, fonte_oficial, nivel_confianca, acao_recomendada }
```

| Arquivo | Responsabilidade |
|---|---|
| [src/config.py](src/config.py) | Configuração via `.env` com validação (threshold, modelos, limites) |
| [src/rag_engine.py](src/rag_engine.py) | Carga/validação do CSV, embeddings em lote, cache em disco, busca por cosseno |
| [src/business_rules.py](src/business_rules.py) | Detecção de sinais e regras R1–R6 (100% determinístico) |
| [src/agent.py](src/agent.py) | Guardrails, prompt, chamada à LLM, schema Pydantic, enforcement, auditoria, CLI |
| [test_scenarios.py](test_scenarios.py) | 78 testes offline + 7 testes `live` opcionais + demonstração |

## Regras de negócio

Precedência de cima para baixo. Quando uma regra se aplica, a decisão **não** depende da LLM.

| Regra | Condição | Decisão | Fonte na base |
|---|---|---|---|
| R1 | Cliente afirma que não autorizou a compra | `AVALIACAO_MANUAL` | Política 4.4 |
| R1 | Fraude no método de pagamento, ou múltiplos reembolsos (texto ou `reembolsos_anteriores_pedido > 0`) | `AVALIACAO_MANUAL` | Fluxo Segurança |
| R2 | "Não chegou, mas consta como entregue" (status divergente) | `AVALIACAO_MANUAL` | Fluxo de Entrega |
| R3 | Desistência **e** falha operacional no mesmo relato (sinais conflitantes) | `AVALIACAO_MANUAL` | 2.1/2.2 + 3.2/4.1 |
| R4 | Falha do restaurante ou do app | `REEMBOLSO_PERMITIDO` | Política 2.1 (+ 3.2 se já despachado) |
| R4 | Falha do entregador (entrega incorreta, extravio) | `REEMBOLSO_PERMITIDO` | Política 2.2 (+ 3.2 se já despachado) |
| R5 | Desistência do cliente com pedido já despachado | `REEMBOLSO_NEGADO` | Política 3.2 / Política 4.1 |
| R6 | Reembolso/cancelamento pedido após o despacho, sem motivo identificado | `AVALIACAO_MANUAL` | Política 3.2 |
| — | Nenhum sinal | decidido pela LLM **somente** com base no contexto recuperado | fonte do contexto |

As fontes vêm exatamente da coluna `fonte` do CSV. Um teste garante que toda fonte usada pelas regras
existe na base.

Os sinais são extraídos por expressões regulares sobre texto normalizado (sem acentos), com tratamento
simples de negação ("não houve falha do restaurante" não dispara R3). Dados estruturados do pedido
(`pedido_despachado`, `reembolsos_anteriores_pedido`) têm precedência sobre o texto.

Valores possíveis de `decisao`: `REEMBOLSO_PERMITIDO`, `REEMBOLSO_NEGADO`, `AVALIACAO_MANUAL`,
`ORIENTACAO` (dúvida informativa, ex.: prazo de estorno) e `TRANSBORDO_HUMANO` (fallback seguro).

## Guardrails

- **Sanitização:** NFKC, remoção de caracteres nulos/de controle/invisíveis (zero-width, bidi override),
  neutralização de `<` `>` (inclusive variantes full-width) e de separadores (`###`, ```` ``` ````, `----`),
  limite de tamanho (`MAX_INPUT_CHARS`).
- **Detecção de prompt injection:** padrões como "ignore as instruções anteriores", troca de papel,
  pedido de revelar o prompt, `SYSTEM:`, tags estruturais e "aprove sem validar". A mensagem é bloqueada
  **antes** do RAG e da LLM.
- **Prompt com tags XML:** contexto e mensagem do usuário isolados em `<contexto>` e `<mensagem_usuario>`;
  todo conteúdo é escapado, então o usuário não consegue fechar ou abrir tags.
- **Saída validada:** `AgentResponse` (Pydantic, `extra="forbid"`, enum de decisões, limites de tamanho).
  `nivel_confianca` é a similaridade de cosseno calculada pelo sistema, nunca um valor declarado pela LLM.
- **Fallback seguro (Fluxo Suporte):** toda resposta sem decisão conclusiva (`AVALIACAO_MANUAL` ou
  `TRANSBORDO_HUMANO`) orienta validação manual ou abertura de ticket interno e inclui o link de suporte
  (`SUPPORT_URL`).
- **Auditoria:** cada resposta gera um log JSON no logger `ifood_agent.audit` (decisão, fonte, regra,
  se a LLM foi chamada, motivo de fallback).

## Como executar

Requer Python 3.10+.

```bash
cd ifood-genai-agent-poc
python -m venv .venv
# Windows: .venv\Scripts\activate    |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # preencha OPENAI_API_KEY
```

**CLI**

```bash
python -m src.agent "O cliente desistiu do pedido depois que saiu para entrega, tem reembolso?"
python -m src.agent "Cliente desistiu do pedido" --despachado --debug
python -m src.agent "Restaurante mandou item errado" --reembolsos-anteriores 1
```

Saída (resumida):

```json
{
  "decisao": "REEMBOLSO_NEGADO",
  "justificativa": "O cliente desistiu após a saída para entrega...",
  "fonte_oficial": "Política 3.2 / Política 4.1",
  "nivel_confianca": 0.8731,
  "acao_recomendada": "Informar ao cliente que o reembolso não é elegível..."
}
```

**Como biblioteca**

```python
from src.agent import RefundAgent

agent = RefundAgent.from_settings()
result = agent.responder("Entregador não entregou o pedido", pedido_despachado=True)
print(result.resposta.model_dump())   # contrato de saída
print(result.to_dict()["metadados"])  # políticas recuperadas, regra, fallback
```

## Testes

```bash
pytest -v                                          # 78 testes offline (sem rede, sem custo)
RUN_LIVE_TESTS=1 pytest -v -m live                 # integração real com a OpenAI
python test_scenarios.py                           # demonstração em tabela
```

Os testes offline usam um embedder fake (bag-of-words) e LLMs fake ("obediente", "rebelde" e
"quebrada") para cobrir threshold e fallback, as regras R1–R6, injeção de prompt, sanitização,
sobrescrita de LLM divergente, fonte inventada, JSON inválido, LLM indisponível, schema Pydantic e
cache de embeddings.

## Configuração

Todas as variáveis estão em [.env.example](.env.example). As principais:

| Variável | Padrão | Observação |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.75` | Abaixo disso não há chamada à LLM |
| `CHAT_MODEL` | `gpt-4.1-mini` | Precisa suportar `response_format=json_object` e `temperature` |
| `TOP_K` | `3` | Políticas recuperadas (só entram no contexto as que passam do threshold) |
| `EMBEDDINGS_CACHE_DIR` | `.cache` | Embeddings da base são recalculados só quando o CSV ou o modelo muda |

### Calibração do threshold

Com `text-embedding-3-small`, paráfrases em português da mesma dúvida costumam ficar entre 0,60 e 0,85
de similaridade. Um limiar de 0,75 é **conservador**: reduz o risco de resposta errada, mas aumenta o
transbordo. Antes de produção, monte um conjunto rotulado (dúvidas reais → política correta / fora de
escopo), rode `RAGEngine.search` e escolha o limiar pela curva de precisão × cobertura. Outra forma de
subir a cobertura sem baixar o limiar é cadastrar variações de pergunta para a mesma política no CSV.

## Limitações e próximos passos para produção

- **Detecção de sinais por regex:** é auditável e previsível, mas tem cobertura limitada. Evolução natural:
  receber `motivo` e `status` estruturados do backoffice, ou um classificador de intenção com saída
  validada, mantendo as regras em código como árbitro final.
- **Ordem threshold × regras:** por requisito, uma dúvida abaixo do limiar vai para o fallback mesmo com
  sinal de fraude. Os dois caminhos levam a análise humana, mas o roteamento direto ao antifraude
  poderia vir antes do threshold.
- **Índice em memória:** adequado para centenas de políticas. Para bases maiores, use um vector store
  (pgvector, OpenSearch) e versione a base de políticas.
- **Observabilidade:** enviar o log de auditoria para a stack de monitoramento, acompanhar a taxa de
  fallback e as divergências entre LLM e regra, e coletar feedback do colaborador.
- **PII:** mascarar dados pessoais (CPF, cartão, telefone) antes de enviar texto a provedores externos.
- **Avaliação contínua:** um golden set de casos com a decisão esperada rodando no CI (os testes `live`
  são o ponto de partida).
- **Integração com APIs de pedido e estorno:** a base indica consultar a API interna para validar o status
  do pedido (Integração API). Hoje isso entra pelos parâmetros `pedido_despachado` e
  `reembolsos_anteriores_pedido`; o próximo passo é uma API fictícia que preencha esses dados pelo ID do
  pedido e resolva divergências entre status financeiro e status do pedido.

## Ideias de evolução do README do iFood: o que já está na POC

| Ideia | Situação |
|---|---|
| Logs de confiança da resposta | Feito: `nivel_confianca` em toda resposta e log JSON de auditoria |
| Categorias de decisão (financeiro / restaurante / entrega / fraude) | Feito: regras R1–R6 e `categoria` das políticas nos metadados |
| Classificação automática do tipo de caso | Feito: detecção de sinais em `business_rules.py` |
| Integração com APIs fictícias de pedido/estorno | Próximo passo (ver acima) |
