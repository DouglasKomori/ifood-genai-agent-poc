"""Regras determinísticas de negócio para reembolso/cancelamento.

Esta camada roda em código (não na LLM). Quando uma regra se aplica, a decisão é
definitiva: a LLM apenas redige a justificativa e o agente rejeita qualquer divergência.

Ordem de precedência (fontes conforme a base de conhecimento):
    R1  Fraude: não autorização (Política 4.4), fraude no pagamento
        ou múltiplos reembolsos (Fluxo Segurança)                  -> AVALIACAO_MANUAL
    R2  Status divergente: "não chegou, mas consta como entregue"
        (Fluxo de Entrega)                                         -> AVALIACAO_MANUAL
    R3  Sinais conflitantes (desistência + falha operacional)      -> AVALIACAO_MANUAL
    R4  Falha do restaurante ou do app (Política 2.1) ou do
        entregador (Política 2.2); após despacho, + Política 3.2   -> REEMBOLSO_PERMITIDO
    R5  Desistência do cliente após despacho (Política 3.2 / 4.1)  -> REEMBOLSO_NEGADO
    R6  Reembolso após despacho sem motivo identificado (3.2)      -> AVALIACAO_MANUAL
    --  Nenhum sinal determinístico                                -> decisão guiada pelo contexto RAG
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum


class Decisao(str, Enum):
    REEMBOLSO_PERMITIDO = "REEMBOLSO_PERMITIDO"
    REEMBOLSO_NEGADO = "REEMBOLSO_NEGADO"
    AVALIACAO_MANUAL = "AVALIACAO_MANUAL"
    ORIENTACAO = "ORIENTACAO"  # resposta informativa baseada na política (sem decisão de reembolso)
    TRANSBORDO_HUMANO = "TRANSBORDO_HUMANO"  # fallback seguro: abrir ticket / especialista


class Sinal(str, Enum):
    FRAUDE_NAO_AUTORIZADO = "FRAUDE_NAO_AUTORIZADO"
    FRAUDE_MULTIPLOS_REEMBOLSOS = "FRAUDE_MULTIPLOS_REEMBOLSOS"
    FRAUDE_PAGAMENTO = "FRAUDE_PAGAMENTO"
    STATUS_DIVERGENTE = "STATUS_DIVERGENTE"
    SOLICITA_REEMBOLSO = "SOLICITA_REEMBOLSO"
    DESISTENCIA_CLIENTE = "DESISTENCIA_CLIENTE"
    FALHA_RESTAURANTE = "FALHA_RESTAURANTE"
    FALHA_APLICATIVO = "FALHA_APLICATIVO"
    FALHA_ENTREGADOR = "FALHA_ENTREGADOR"
    PEDIDO_DESPACHADO = "PEDIDO_DESPACHADO"


# Fontes exatamente como aparecem na coluna `fonte` do CSV.
FONTE_FALHA_RESTAURANTE = "Política 2.1"   # "Erro do restaurante garante reembolso?"
FONTE_FALHA_APP = "Política 2.1"           # "Cancelamento por erro do app."
FONTE_FALHA_ENTREGADOR = "Política 2.2"    # "Falha do entregador pode gerar reembolso?"
FONTE_POS_DESPACHO = "Política 3.2"        # "O cliente pode pedir reembolso após o pedido sair para entrega?"
FONTE_DESISTENCIA = ("Política 3.2", "Política 4.1")
FONTE_NAO_AUTORIZADO = "Política 4.4"
FONTE_SEGURANCA = "Fluxo Segurança"
FONTE_STATUS_ENTREGA = "Fluxo de Entrega"

_SINAIS_FRAUDE = frozenset({
    Sinal.FRAUDE_NAO_AUTORIZADO, Sinal.FRAUDE_PAGAMENTO, Sinal.FRAUDE_MULTIPLOS_REEMBOLSOS,
})
_SINAIS_FALHA = frozenset({Sinal.FALHA_RESTAURANTE, Sinal.FALHA_APLICATIVO, Sinal.FALHA_ENTREGADOR})


@dataclass(frozen=True)
class RuleDecision:
    regra: str
    decisao: Decisao
    fontes: tuple[str, ...]
    justificativa: str
    acao_recomendada: str
    sinais: frozenset[Sinal]

    @property
    def fonte_oficial(self) -> str:
        return " / ".join(self.fontes)


def normalize_text(text: str) -> str:
    """Minúsculas, sem acentos e com espaços colapsados — base para os padrões abaixo."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", without_accents).strip()


# Padrões aplicados sobre texto normalizado (sem acentos, minúsculo).
_W = r"(?:\w+ )"  # uma palavra seguida de espaço, para janelas curtas entre termos
_W_SEM_LOCAL = r"(?:(?!endereco\b|local\b)\w+ )"  # idem, sem atravessar "endereço" ("pedido no endereço errado")
_PATTERNS: dict[Sinal, tuple[str, ...]] = {
    Sinal.FRAUDE_NAO_AUTORIZADO: (
        rf"\bnao {_W}{{0,2}}autoriz(?:ou|ei|ado|ada|aram|amos)\b",
        r"\bnao reconhec\w*",
        r"\bcompra (?:nao reconhecida|desconhecida)\b",
        r"\bsem (?:a |sua |minha )?autorizacao\b",
    ),
    Sinal.FRAUDE_PAGAMENTO: (
        r"\bcartao (?:foi )?clonado\b",
        r"\bconta (?:foi )?(?:invadida|hackeada|roubada)\b",
        rf"\bfraude {_W}{{0,3}}(?:pagamento|cartao|pix)\b",
    ),
    Sinal.SOLICITA_REEMBOLSO: (
        r"\breembols\w*",
        r"\bestorn\w*",
        r"\bcancel\w*",
        r"\bdinheiro de volta\b",
        r"\bdevoluc\w*",
    ),
    Sinal.STATUS_DIVERGENTE: (
        rf"\b(?:consta|aparece|constando|aparecendo|marcad[oa]|registrad[oa]|esta) {_W}{{0,2}}como entregue\b",
        r"\bstatus (?:de |como )?entregue\b",
    ),
    Sinal.FRAUDE_MULTIPLOS_REEMBOLSOS: (
        r"\b(?:multiplos|multiplas|varios|varias|diversos|diversas|sucessivos|repetidos) "
        r"(?:pedidos de |solicitacoes de )?reembolsos?\b",
        r"\bmais de (?:um|uma|dois|duas|\d+) (?:vez(?:es)? |pedidos? de |solicitac\w+ de )?reembolsos?\b",
        r"\breembolsos? (?:\w+ ){0,1}mais de (?:uma|duas|\d+) vez",
        r"\breembolsos? (?:de novo|novamente|outra vez|repetid\w*|duplicad\w*)\b",
        r"\b(?:segundo|terceiro|outro) reembolso\b",
        r"\bja (?:recebeu|ganhou|teve|obteve) (?:o |um )?(?:reembolso|estorno)\b",
        r"\bja foi reembolsad\w*",
    ),
    Sinal.DESISTENCIA_CLIENTE: (
        r"\bdesist\w*",
        r"\bmudou de ideia\b",
        r"\bnao (?:quer|deseja|precisa) mais\b",
        r"\barrepend\w*",
    ),
    Sinal.FALHA_RESTAURANTE: (
        r"\b(?:falha|erro) (?:comprovad[oa] )?do restaurante\b",
        r"\brestaurante (?:errou|falhou|esqueceu|trocou)\b",
        rf"\brestaurante {_W}{{0,4}}(?:errad[oa]|incomplet[oa]|trocad[oa])\b",
        rf"\b(?:pedido|item|itens|produto|lanche|prato) {_W_SEM_LOCAL}{{0,2}}"
        r"(?:errad[oa]s?|trocad[oa]s?|faltando|faltantes?|incomplet[oa]s?|estragad[oa]s?|vencid[oa]s?)\b",
        r"\bfalt(?:ou|aram|ando) (?:um |uma |o |a |os |as )?(?:item|itens|produto|lanche|bebida)",
        rf"\b(?:comida|alimento) {_W}{{0,2}}(?:estragad|azed|podre)\w*",
        r"\bcorpo estranho\b",
    ),
    Sinal.FALHA_APLICATIVO: (
        rf"\b(?:falha|erro|bug|problema|instabilidade) {_W}{{0,2}}(?:do|no|da|na) "
        r"(?:app|aplicativo|plataforma|sistema)\b",
        rf"\b(?:app|aplicativo|plataforma|sistema) {_W}{{0,1}}(?:travou|caiu|falhou|bugou)\b",
    ),
    Sinal.FALHA_ENTREGADOR: (
        rf"\b(?:falha|erro|problema) {_W}{{0,1}}(?:do|da) (?:entregador|entregadora|motoboy|parceiro de entrega)\b",
        rf"\b(?:entregador|entregadora|motoboy) {_W}{{0,3}}"
        r"(?:nao entregou|derrubou|extraviou|perdeu|abriu|violou|sumiu|comeu)\b",
        rf"\bentreg(?:ue|ou) {_W}{{0,2}}(?:no|em) endereco errado\b",
        r"\bpedido (?:foi )?extraviad[oa]\b",
        r"\bembalagem (?:foi )?(?:violada|aberta)\b",
    ),
    Sinal.PEDIDO_DESPACHADO: (
        r"\bsaiu para (?:a )?entrega\b",
        r"\bsaida (?:do pedido )?para (?:a )?entrega\b",
        r"\bdespachad\w*",
        r"\b(?:apos|depois do|depois de) (?:o )?despacho\b",
        r"\ba caminho\b",
        r"\bem rota\b",
        r"\bcom o entregador\b",
        r"\b(?:depois|apos) (?:que |de )?(?:o pedido )?(?:saiu|sair)\b",
        r"\bcoletad[oa] pelo entregador\b",
    ),
}

_COMPILED: dict[Sinal, tuple[re.Pattern[str], ...]] = {
    sinal: tuple(re.compile(p) for p in patterns) for sinal, patterns in _PATTERNS.items()
}

# Negação imediatamente antes do termo ("não houve falha do restaurante", "sem erro no app").
# Padrões que já contêm a negação (ex.: "não autorizou") começam no "nao" e não são afetados.
_NEGATION_BEFORE = re.compile(r"\b(?:nao|sem|nenhum\w*|nunca|jamais)\b(?: \w+){0,2} ?$")


# Sinais cujo relato típico já traz uma negação antes do termo ("não chegou, mas consta como entregue").
_SEM_CHECAGEM_DE_NEGACAO = frozenset({Sinal.STATUS_DIVERGENTE, Sinal.SOLICITA_REEMBOLSO})


def _matches(text: str, patterns: tuple[re.Pattern[str], ...], check_negation: bool = True) -> bool:
    for pattern in patterns:
        for match in pattern.finditer(text):
            if not check_negation or not _NEGATION_BEFORE.search(text[: match.start()]):
                return True
    return False


def detectar_sinais(
    texto: str,
    *,
    pedido_despachado: bool | None = None,
    reembolsos_anteriores_pedido: int = 0,
) -> frozenset[Sinal]:
    """Extrai sinais de negócio do texto + dados estruturados do pedido (que têm precedência)."""
    normalized = normalize_text(texto)
    sinais = {
        sinal for sinal, patterns in _COMPILED.items()
        if _matches(normalized, patterns, check_negation=sinal not in _SEM_CHECAGEM_DE_NEGACAO)
    }

    if pedido_despachado is True:
        sinais.add(Sinal.PEDIDO_DESPACHADO)
    elif pedido_despachado is False:
        sinais.discard(Sinal.PEDIDO_DESPACHADO)

    if reembolsos_anteriores_pedido > 0:
        sinais.add(Sinal.FRAUDE_MULTIPLOS_REEMBOLSOS)
    return frozenset(sinais)


_ACAO_ANTIFRAUDE = (
    "Não conceder reembolso. Encaminhar o caso para avaliação manual da equipe Antifraude "
    "(Fluxo Segurança) com ID do pedido, histórico de reembolsos e relato do cliente; "
    "aplicar bloqueio preventivo se houver indício de acesso indevido."
)


def avaliar_regras(
    texto: str,
    *,
    pedido_despachado: bool | None = None,
    reembolsos_anteriores_pedido: int = 0,
) -> RuleDecision | None:
    """Aplica as regras determinísticas. Retorna None quando nenhuma regra se aplica."""
    sinais = detectar_sinais(
        texto,
        pedido_despachado=pedido_despachado,
        reembolsos_anteriores_pedido=reembolsos_anteriores_pedido,
    )
    fraude = sinais & _SINAIS_FRAUDE
    falhas = sinais & _SINAIS_FALHA

    # R1 — Fraude / exceção de segurança
    if fraude:
        fontes: list[str] = []
        motivos: list[str] = []
        if Sinal.FRAUDE_NAO_AUTORIZADO in fraude:
            fontes.append(FONTE_NAO_AUTORIZADO)
            motivos.append("o cliente afirma não ter autorizado/reconhecido a compra")
        if Sinal.FRAUDE_PAGAMENTO in fraude:
            fontes.append(FONTE_SEGURANCA)
            motivos.append("há suspeita de fraude no método de pagamento")
        if Sinal.FRAUDE_MULTIPLOS_REEMBOLSOS in fraude:
            fontes.append(FONTE_SEGURANCA)
            motivos.append("há indício de múltiplas solicitações de reembolso para o mesmo pedido")
        return RuleDecision(
            regra="R1_FRAUDE_EXCECAO",
            decisao=Decisao.AVALIACAO_MANUAL,
            fontes=_unique(fontes),
            justificativa=(
                f"Caso com sinal de risco: {' e '.join(motivos)}. Pela política, situações de fraude "
                "ou exceção não admitem reembolso direto e exigem avaliação manual."
            ),
            acao_recomendada=_ACAO_ANTIFRAUDE,
            sinais=sinais,
        )

    # R2 — Status divergente: consta como entregue, mas o cliente não recebeu
    if Sinal.STATUS_DIVERGENTE in sinais:
        return RuleDecision(
            regra="R2_STATUS_DIVERGENTE",
            decisao=Decisao.AVALIACAO_MANUAL,
            fontes=(FONTE_STATUS_ENTREGA,),
            justificativa=(
                "O status do pedido (entregue) diverge do relato do cliente. Pela política, esse caso "
                "exige validação manual antes de qualquer decisão; o reembolso só é possível após análise."
            ),
            acao_recomendada=(
                "Validar o status real na API interna de pedidos (rastreamento, geolocalização e "
                "comprovante de entrega). Se a falha de entrega se confirmar, seguir a Política 2.2; "
                "se a divergência persistir, abrir ticket interno com as evidências."
            ),
            sinais=sinais,
        )

    # R3 — Sinais conflitantes: não decidir automaticamente
    if Sinal.DESISTENCIA_CLIENTE in sinais and falhas:
        return RuleDecision(
            regra="R3_SINAIS_CONFLITANTES",
            decisao=Decisao.AVALIACAO_MANUAL,
            fontes=_unique([*_fontes_falha(falhas), *FONTE_DESISTENCIA]),
            justificativa=(
                "O relato combina desistência do cliente com indício de falha operacional. "
                "As políticas levam a decisões opostas nesses casos, então a causa precisa ser "
                "apurada antes de qualquer decisão."
            ),
            acao_recomendada=(
                "Apurar a causa principal (rastreamento, chat com restaurante/entregador e evidências). "
                "Se a falha for comprovada, aplicar Política 2.1/2.2; se for desistência após o "
                "despacho, aplicar Política 3.2/4.1. Em caso de dúvida, abrir ticket interno."
            ),
            sinais=sinais,
        )

    # R4 — Falha da operação: reembolso elegível
    if falhas:
        responsaveis = [
            nome for sinal, nome in (
                (Sinal.FALHA_RESTAURANTE, "restaurante"),
                (Sinal.FALHA_APLICATIVO, "aplicativo"),
                (Sinal.FALHA_ENTREGADOR, "entregador"),
            ) if sinal in falhas
        ]
        fontes_falha = _fontes_falha(falhas)
        if Sinal.PEDIDO_DESPACHADO in sinais:
            fontes_falha.append(FONTE_POS_DESPACHO)
        return RuleDecision(
            regra="R4_FALHA_OPERACIONAL",
            decisao=Decisao.REEMBOLSO_PERMITIDO,
            fontes=_unique(fontes_falha),
            justificativa=(
                f"Falha atribuída ao {' / '.join(responsaveis)}. Pela política, falhas comprovadas do "
                "restaurante, do aplicativo ou do entregador tornam o reembolso elegível."
            ),
            acao_recomendada=(
                "Validar a evidência da falha (foto, registro no chat ou incidente técnico) e, "
                "confirmada, efetuar o reembolso no meio de pagamento original, registrando a "
                "ocorrência contra o responsável."
            ),
            sinais=sinais,
        )

    # R5 — Desistência após despacho: não elegível
    if Sinal.DESISTENCIA_CLIENTE in sinais and Sinal.PEDIDO_DESPACHADO in sinais:
        return RuleDecision(
            regra="R5_DESISTENCIA_POS_DESPACHO",
            decisao=Decisao.REEMBOLSO_NEGADO,
            fontes=FONTE_DESISTENCIA,
            justificativa=(
                "O cliente desistiu do pedido após a saída para entrega. Pela política, após o despacho "
                "o reembolso só cabe com falha do restaurante, do app ou do entregador; a desistência "
                "do cliente não é elegível."
            ),
            acao_recomendada=(
                "Confirmar no histórico do pedido que não houve falha do restaurante, do app ou do "
                "entregador. Sem falha, informar ao cliente que o reembolso não é elegível e registrar o "
                "atendimento; havendo qualquer indício de falha, encaminhar para validação manual."
            ),
            sinais=sinais,
        )

    # R6 — Pedido de reembolso após despacho sem motivo identificado: a política depende do motivo
    if Sinal.SOLICITA_REEMBOLSO in sinais and Sinal.PEDIDO_DESPACHADO in sinais:
        return RuleDecision(
            regra="R6_MOTIVO_NAO_IDENTIFICADO",
            decisao=Decisao.AVALIACAO_MANUAL,
            fontes=(FONTE_POS_DESPACHO,),
            justificativa=(
                "O pedido já saiu para entrega e o motivo da solicitação não foi informado. Pela política, "
                "após o despacho o reembolso só cabe com falha do restaurante, do app ou do entregador; "
                "a desistência do cliente pode não ser elegível. Sem o motivo, não é possível decidir."
            ),
            acao_recomendada=(
                "Confirmar o motivo com o cliente e o status na API interna de pedidos. Falha do "
                "restaurante ou do app: Política 2.1 (elegível). Falha do entregador: Política 2.2 "
                "(elegível). Desistência do cliente: Política 3.2/4.1 (não elegível). Persistindo a "
                "dúvida, abrir ticket interno."
            ),
            sinais=sinais,
        )

    return None


def _fontes_falha(falhas: frozenset[Sinal]) -> list[str]:
    fontes = []
    if Sinal.FALHA_RESTAURANTE in falhas:
        fontes.append(FONTE_FALHA_RESTAURANTE)
    if Sinal.FALHA_APLICATIVO in falhas:
        fontes.append(FONTE_FALHA_APP)
    if Sinal.FALHA_ENTREGADOR in falhas:
        fontes.append(FONTE_FALHA_ENTREGADOR)
    return fontes


def _unique(fontes: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fontes))
