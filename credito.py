"""
🏍️ Simulador de Aprovação de Crédito — Financiamento de Motocicletas
Motor de decisão em TensorFlow + interface visual em Streamlit

Regras de negócio implementadas:
1. Idade mínima: 18 anos (abaixo disso, negativa automática por menoridade)
2. Comprometimento de renda: parcela não pode ultrapassar 30% da renda líquida
3. Aprovado SE parcela <= limite E idade >= 18
4. Negado por parcela alta SE parcela > limite E idade >= 18 (sugere nova parcela)
5. Renda <= 0 ou não informada: pede esclarecimento, não calcula
6. Sempre exibe o cálculo (transparência) antes da decisão
"""

import math
import streamlit as st
import tensorflow as tf


# ---------------------------------------------------------------------------
# MOTOR DE DECISÃO (TensorFlow)
# ---------------------------------------------------------------------------
# Mesmo sendo regras determinísticas (não um modelo treinado), a lógica
# matemática central roda sobre tensores do TensorFlow, para que todo o
# cálculo de comprometimento de renda e as comparações de aprovação
# aconteçam dentro do grafo/execução do TF (tf.function).

PERCENTUAL_COMPROMETIMENTO = 0.30  # 30% da renda líquida
IDADE_MINIMA = 18


@tf.function
def calcular_limite_parcela(renda: tf.Tensor) -> tf.Tensor:
    """limite_parcela = renda * 0.30"""
    return tf.multiply(renda, tf.constant(PERCENTUAL_COMPROMETIMENTO, dtype=tf.float32))


@tf.function
def avaliar_credito(idade: tf.Tensor, renda: tf.Tensor, parcela: tf.Tensor):
    """
    Retorna um dicionário de tensores com:
    - limite_parcela
    - idade_valida (bool)
    - parcela_ok (bool)
    - aprovado (bool)
    """
    limite_parcela = calcular_limite_parcela(renda)
    idade_valida = tf.greater_equal(idade, tf.constant(IDADE_MINIMA, dtype=tf.float32))
    parcela_ok = tf.less_equal(parcela, limite_parcela)
    aprovado = tf.logical_and(idade_valida, parcela_ok)

    return {
        "limite_parcela": limite_parcela,
        "idade_valida": idade_valida,
        "parcela_ok": parcela_ok,
        "aprovado": aprovado,
    }


def rodar_motor_decisao(idade: float, renda: float, parcela: float) -> dict:
    """Wrapper Python que converte os inputs do formulário em tensores,
    chama o motor TensorFlow e devolve valores nativos (float/bool) prontos
    para exibição no Streamlit."""

    idade_t = tf.constant(idade, dtype=tf.float32)
    renda_t = tf.constant(renda, dtype=tf.float32)
    parcela_t = tf.constant(parcela, dtype=tf.float32)

    resultado = avaliar_credito(idade_t, renda_t, parcela_t)

    return {
        "limite_parcela": float(resultado["limite_parcela"].numpy()),
        "idade_valida": bool(resultado["idade_valida"].numpy()),
        "parcela_ok": bool(resultado["parcela_ok"].numpy()),
        "aprovado": bool(resultado["aprovado"].numpy()),
    }


def sugerir_parcela(renda: float) -> float:
    """Sugestão de parcela = renda * 0.30, arredondada para baixo, 2 casas decimais."""
    valor = renda * PERCENTUAL_COMPROMETIMENTO
    return math.floor(valor * 100) / 100


# ---------------------------------------------------------------------------
# FORMATAÇÃO (parecer de analista)
# ---------------------------------------------------------------------------

def formatar_moeda(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def montar_parecer(idade: float, renda: float | None, parcela: float, resultado: dict | None) -> str:
    linhas = []

    renda_str = formatar_moeda(renda) if renda is not None else "não informada"
    linhas.append(
        f"📋 **Dados recebidos:** Idade: {int(idade)} anos | "
        f"Renda: {renda_str} | Parcela solicitada: {formatar_moeda(parcela)}"
    )

    # Regra 5 — renda insuficiente / não informada
    if renda is None or renda <= 0:
        linhas.append("🧮 **Cálculo:** não é possível calcular o limite de parcela sem uma renda válida.")
        linhas.append("⚠️ **Resultado:** Esclarecimento necessário")
        linhas.append(
            "💬 **Justificativa:** a renda mensal líquida informada é inválida ou não foi preenchida. "
            "Por favor, informe um valor de renda maior que zero para prosseguir com a simulação."
        )
        return "\n\n".join(linhas)

    # Regra 1 — menoridade (negativa automática, sem cálculo de parcela)
    if idade < IDADE_MINIMA:
        linhas.append("🧮 **Cálculo:** não aplicável — idade abaixo do mínimo exigido, sem sugestão de parcela.")
        linhas.append("❌ **Resultado:** Negado")
        linhas.append(
            f"💬 **Justificativa:** a idade mínima para financiamento é de {IDADE_MINIMA} anos. "
            "Solicitante menor de idade não pode contratar o financiamento."
        )
        return "\n\n".join(linhas)

    # Regras 2, 3 e 4 — comprometimento de renda
    limite = resultado["limite_parcela"]
    linhas.append(
        f"🧮 **Cálculo:** 30% da renda = {formatar_moeda(renda)} × 0,30 = **{formatar_moeda(limite)}**"
    )

    if resultado["aprovado"]:
        linhas.append("✅ **Resultado:** Aprovado")
        linhas.append(
            f"💬 **Justificativa:** a parcela solicitada ({formatar_moeda(parcela)}) está dentro do limite de "
            f"30% da renda líquida ({formatar_moeda(limite)}). Crédito aprovado."
        )
    else:
        sugestao = sugerir_parcela(renda)
        linhas.append("❌ **Resultado:** Negado por comprometimento de renda")
        linhas.append(
            f"💬 **Justificativa:** a parcela solicitada ({formatar_moeda(parcela)}) ultrapassa o limite de "
            f"30% da renda líquida ({formatar_moeda(limite)})."
        )
        linhas.append(f"💡 **Sugestão:** Parcela recomendada: {formatar_moeda(sugestao)}")

    return "\n\n".join(linhas)


# ---------------------------------------------------------------------------
# INTERFACE STREAMLIT
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Simulador de Crédito — Motos", page_icon="🏍️", layout="centered")

st.title("🏍️ Simulador de Aprovação de Crédito")
st.caption("Financiamento de motocicletas — motor de decisão em TensorFlow")

with st.form("form_credito"):
    col1, col2 = st.columns(2)
    with col1:
        idade = st.number_input("Idade (anos)", min_value=0, max_value=120, value=22, step=1)
    with col2:
        renda = st.number_input(
            "Renda mensal líquida (R$)", min_value=0.0, value=2000.0, step=100.0, format="%.2f"
        )

    parcela = st.number_input(
        "Valor da parcela pretendida (R$)", min_value=0.0, value=600.0, step=50.0, format="%.2f"
    )

    enviado = st.form_submit_button("Simular crédito", use_container_width=True)

if enviado:
    st.divider()

    renda_valida = renda if renda > 0 else None
    resultado = None

    if renda_valida is not None and idade >= IDADE_MINIMA:
        resultado = rodar_motor_decisao(float(idade), float(renda), float(parcela))

    parecer = montar_parecer(float(idade), renda_valida, float(parcela), resultado)

    # Cor de destaque conforme o desfecho
    if renda_valida is None:
        st.warning(parecer)
    elif idade < IDADE_MINIMA:
        st.error(parecer)
    elif resultado["aprovado"]:
        st.success(parecer)
    else:
        st.error(parecer)

    with st.expander("🔍 Detalhes técnicos do motor de decisão (TensorFlow)"):
        st.write(f"- TensorFlow versão: `{tf.__version__}`")
        st.write(f"- Percentual de comprometimento de renda: `{PERCENTUAL_COMPROMETIMENTO * 100:.0f}%`")
        st.write(f"- Idade mínima exigida: `{IDADE_MINIMA} anos`")
        if resultado is not None:
            st.json(resultado)
else:
    st.info("Preencha o formulário acima e clique em **Simular crédito** para ver o parecer.")
