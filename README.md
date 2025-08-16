# Credit Card Fraud Detection com Keras

## **👤 Integrante:**

* [Izadora Luz](https://www.linkedin.com/in/izadoraluz-rsn/)

## **👨‍🏫 Professores:**

* [Cesar Almiñana](https://www.linkedin.com/in/ccalminana/) - Professor orientador
* [Crishna Irion](https://www.linkedin.com/in/crishna-irion-7b5aa311/) - Professora de redes neurais

## **📝 Descrição**

Este projeto tem como objetivo desenvolver e avaliar um **modelo de classificação binária** utilizando o dataset público **Credit Card Fraud Detection (Kaggle – MLG-ULB)**. O dataset contém mais de **280 mil transações**, das quais apenas **492 são fraudes** (\~0,17%), caracterizando um **problema altamente desbalanceado**.

O modelo escolhido foi um **Sequential em Keras** com **uma única camada Dense com ativação sigmoid**, equivalente a uma regressão logística, sendo treinado para classificar transações como **fraude (1)** ou **legítima (0)**.

### Funcionalidades implementadas:

* **Pré-processamento:** Padronização das variáveis `Time` e `Amount`, mantendo os componentes principais (V1–V28) já normalizados via PCA.
* **Divisão dos dados:** Separação em treino e teste com estratificação, para manter a proporção das classes.
* **Modelo:** Rede neural sequencial com uma camada `Dense(1, activation="sigmoid")`.
* **Compilação:** `optimizer=adam`, `loss=binary_crossentropy`, métricas `accuracy` e **F1-score** (implementada como métrica custom).
* **Treinamento:** 50 épocas, batch size = 10, com `class_weight` para lidar com o desbalanceamento.
* **Avaliação:** Predição no conjunto de teste e cálculo de **accuracy** e **F1-score**.

## **📁 Estrutura de pastas**

* `src`
    * `data/`

        * `creditcard.csv`: Dataset de transações de cartão de crédito (Kaggle).
        * `creditcardfraud.zip`: Dataset original de transações de cartão de crédito (Kaggle).

  * `fraud_detection.ipynb`: Notebook com o código completo do projeto.
* `README.md`: Documentação do projeto.

## **💻 Tecnologias Utilizadas**

* **Python 3.9**
* **TensorFlow / Keras** (modelo sequencial)
* **Scikit-learn** (pré-processamento, métricas e split)
* **Pandas & NumPy** (manipulação de dados)

## **🧪 Experimentos:**

1. **Treino do modelo com `class_weight`:**

   * O uso de `class_weight` equilibra o impacto das classes no gradiente, permitindo que a rede preste mais atenção na classe minoritária (fraude).

2. **Avaliação com métricas complementares:**

   * A **accuracy** tende a ser alta (>99%) mas pouco informativa em datasets desbalanceados.
   * O **F1-score** fornece uma avaliação mais justa, equilibrando *precision* e *recall* da classe fraude.

3. **Possíveis melhorias futuras:**

   * Ajuste do threshold de decisão para otimizar recall ou F1.
   * Técnicas de reamostragem (SMOTE, undersampling) para reduzir o desbalanceamento.
   * Uso de métricas adicionais como **ROC-AUC** e **AUC-PR**.
   * Teste de arquiteturas mais profundas com dropout e regularização.

