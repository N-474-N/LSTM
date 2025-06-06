# TECH CHALLENGE - FASE 4 - Previsão da bolsa de valores

## Requisitos para executar a API

  1. Python 3.11 ou superior: necessário para rodar a API localmente e criar o pacote para a Lambda.;
  2. Instalação de pacotes de acordo com o requirements.txt disponível;
  3. Conexão com internet para coleta de dados via Yahoo Finance.


## Estrutura:

  - **main**: Arquivo fonte da API construída em FastAPI, especificação da uma classe de retorno, parâmetros da payload, chamada da função de treinamento (LSTM.py) e exposição do endpoint;
    
  - **LSTM** (function): Esse script é responsável pelo treinamento de modelos específicos para cada ação(papel) utilizando o período fixo de 10 meses (a partir da data de execução atual), cada chamada da API aciona essa função que treina o modelo e o salva no diretório local de execução, caso haja uma nova chamada o modelo não é treinado novamente, a versão local (treinada anteriormente) é utilizada
 
  
## Endpoints:

### POST /predict
  * **Descrição:** Executa treinamento do modelo e gera previsão do próximo fechamento
  * **Resposta:** JSON com ação e valor do próxima fechamento
  * **Modelo de** Resposta: (clase) Papel
    
### GET /
  * **Descrição:** Landing page (apenas para debug)
  * **Resposta:** JSON (texto teste)
  * **Modelo de** Resposta: NA
