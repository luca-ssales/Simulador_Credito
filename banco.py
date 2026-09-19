import math

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Simulador de Crédito",
    page_icon="🏍️",
    layout="centered"
)


# =========================================================
# ESTILO
# =========================================================

st.markdown("""
<style>

    .stApp {
        background: #f5f6f8;
    }

    .block-container {
        max-width: 760px;
        padding-top: 40px;
        padding-bottom: 60px;
    }

    .titulo {
        font-size: 32px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 5px;
    }

    .subtitulo {
        color: #6b7280;
        margin-bottom: 30px;
    }

    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
    }

    .aprovado {
        border-left: 5px solid #16a34a;
    }

    .negado {
        border-left: 5px solid #dc2626;
    }

    .calculo {
        background: #f9fafb;
        border-radius: 8px;
        padding: 15px;
        font-family: monospace;
        margin: 15px 0;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def formatar_dinheiro(valor):
    """
    Converte:
    1500.50

    para:
    R$ 1.500,50
    """

    valor_formatado = f"{valor:,.2f}"

    valor_formatado = (
        valor_formatado
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    return f"R$ {valor_formatado}"


def arredondar_para_baixo(valor):
    """
    Arredonda para baixo mantendo duas casas decimais.

    Exemplo:
    633.999 -> 633.99
    """

    return math.floor(valor * 100) / 100


# =========================================================
# MODELO TENSORFLOW
# =========================================================

@st.cache_resource
def criar_modelo():

    # Mantém os resultados mais consistentes
    np.random.seed(42)
    tf.random.set_seed(42)

    quantidade_dados = 15000

    # -----------------------------------------------------
    # GERANDO DADOS DE TREINAMENTO
    # -----------------------------------------------------

    idades = np.random.randint(
        16,
        81,
        quantidade_dados
    )

    rendas = np.random.uniform(
        500,
        50000,
        quantidade_dados
    )

    # Gera parcelas entre 0% e 80% da renda
    porcentagem_parcela = np.random.uniform(
        0,
        0.80,
        quantidade_dados
    )

    parcelas = rendas * porcentagem_parcela


    # -----------------------------------------------------
    # CRIANDO OS RESULTADOS ESPERADOS
    # -----------------------------------------------------

    # 1 = aprovado
    # 0 = negado

    aprovado = (
        (idades >= 18) &
        (parcelas <= rendas * 0.30)
    ).astype(int)


    # -----------------------------------------------------
    # PREPARANDO AS FEATURES
    # -----------------------------------------------------

    proporcao_parcela = parcelas / rendas

    X = np.column_stack([
        idades / 100,
        rendas / 50000,
        proporcao_parcela
    ])

    y = aprovado


    # -----------------------------------------------------
    # CRIANDO A REDE NEURAL
    # -----------------------------------------------------

    modelo = keras.Sequential([

        keras.Input(shape=(3,)),

        layers.Dense(
            16,
            activation="relu"
        ),

        layers.Dense(
            8,
            activation="relu"
        ),

        layers.Dense(
            1,
            activation="sigmoid"
        )

    ])


    # -----------------------------------------------------
    # COMPILANDO
    # -----------------------------------------------------

    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    # -----------------------------------------------------
    # TREINAMENTO
    # -----------------------------------------------------

    modelo.fit(
        X,
        y,
        epochs=20,
        batch_size=32,
        verbose=0
    )

    return modelo


# Carrega / treina apenas uma vez
modelo = criar_modelo()


# =========================================================
# FUNÇÃO PARA PREVISÃO DA REDE NEURAL
# =========================================================

def previsao_tensorflow(idade, renda, parcela):

    proporcao = parcela / renda

    dados = np.array([[
        idade / 100,
        renda / 50000,
        proporcao
    ]])

    previsao = modelo.predict(
        dados,
        verbose=0
    )[0][0]

    return float(previsao)


# =========================================================
# CABEÇALHO
# =========================================================

st.markdown(
    '<div class="titulo">Simulador de Crédito</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitulo">
        Simule o financiamento da sua motocicleta com base
        no comprometimento da sua renda mensal.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FORMULÁRIO
# =========================================================

with st.form("formulario_credito"):

    idade = st.number_input(
        "Idade",
        min_value=0,
        max_value=100,
        step=1,
        placeholder="Ex: 21"
    )

    renda = st.number_input(
        "Renda mensal líquida",
        min_value=0.0,
        step=100.0,
        format="%.2f",
        placeholder="Ex: 2500"
    )

    parcela = st.number_input(
        "Valor da parcela pretendida",
        min_value=0.0,
        step=50.0,
        format="%.2f",
        placeholder="Ex: 700"
    )

    analisar = st.form_submit_button(
        "Simular crédito",
        use_container_width=True
    )


# =========================================================
# PROCESSAMENTO
# =========================================================

if analisar:

    # =====================================================
    # REGRA 5
    # RENDA NÃO INFORMADA OU INVÁLIDA
    # =====================================================

    if renda <= 0:

        st.warning(
            "Informe uma renda mensal líquida maior que R$ 0,00 "
            "para realizar a simulação."
        )

        st.stop()


    # =====================================================
    # CÁLCULO DO LIMITE
    # =====================================================

    limite_parcela = renda * 0.30

    limite_recomendado = arredondar_para_baixo(
        limite_parcela
    )


    # =====================================================
    # DADOS RECEBIDOS
    # =====================================================

    st.markdown("### 📋 Dados recebidos")

    coluna1, coluna2, coluna3 = st.columns(3)

    with coluna1:
        st.metric(
            "Idade",
            f"{idade} anos"
        )

    with coluna2:
        st.metric(
            "Renda",
            formatar_dinheiro(renda)
        )

    with coluna3:
        st.metric(
            "Parcela",
            formatar_dinheiro(parcela)
        )


    # =====================================================
    # TRANSPARÊNCIA DO CÁLCULO
    # =====================================================

    st.markdown("### 🧮 Cálculo")

    st.markdown(
        f"""
        <div class="calculo">
            Limite = renda × 30%<br><br>
            {formatar_dinheiro(renda)} × 0,30<br><br>
            Limite da parcela = <strong>
            {formatar_dinheiro(limite_parcela)}
            </strong>
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # TENSORFLOW
    # =====================================================

    probabilidade = previsao_tensorflow(
        idade,
        renda,
        parcela
    )


    # =====================================================
    # REGRA 1
    # MENOR DE IDADE
    # =====================================================

    if idade < 18:

        st.markdown(
            f"""
            <div class="card negado">

                <h3>❌ Resultado: Negado</h3>

                <p>
                    <strong>Justificativa:</strong>
                    O financiamento exige idade mínima
                    de 18 anos.
                </p>

                <p>
                    Como o solicitante possui
                    <strong>{idade} anos</strong>,
                    a solicitação não pode ser aprovada.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # REGRA 3
    # APROVADO
    # =====================================================

    elif parcela <= limite_parcela:

        st.markdown(
            f"""
            <div class="card aprovado">

                <h3>✅ Resultado: Aprovado</h3>

                <p>
                    <strong>Justificativa:</strong>
                    A parcela solicitada de
                    <strong>{formatar_dinheiro(parcela)}</strong>
                    está dentro do limite de 30% da renda.
                </p>

                <p>
                    Limite permitido:
                    <strong>
                    {formatar_dinheiro(limite_parcela)}
                    </strong>.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # REGRA 4
    # PARCELA ACIMA DO LIMITE
    # =====================================================

    else:

        st.markdown(
            f"""
            <div class="card negado">

                <h3>❌ Resultado: Negado</h3>

                <p>
                    <strong>Justificativa:</strong>
                    A parcela solicitada de
                    <strong>{formatar_dinheiro(parcela)}</strong>
                    ultrapassa o limite de 30% da sua
                    renda mensal.
                </p>

                <p>
                    💡 <strong>Parcela recomendada:</strong>
                    {formatar_dinheiro(limite_recomendado)}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # INFORMAÇÕES DO MODELO
    # =====================================================

    with st.expander(
        "🧠 Ver análise da rede neural"
    ):

        st.write(
            "A rede neural foi criada com TensorFlow/Keras "
            "e treinada com exemplos gerados a partir das "
            "regras de crédito."
        )

        st.write(
            f"Probabilidade estimada de aprovação: "
            f"**{probabilidade * 100:.2f}%**"
        )

        st.caption(
            "A previsão da rede neural é demonstrativa. "
            "A decisão final utiliza as regras matemáticas "
            "definidas pelo banco."
        )