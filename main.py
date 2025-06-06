from functions.LSTM import realiza_treinamento
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
app = FastAPI()

class Papel(BaseModel):
    papel: str

@app.get("/")
def read_root():
    return {"API": "Previsao de fechamento de acoes"}

@app.post("/predict/")
def predicao(papel: Papel):
    valor_previsto = realiza_treinamento(papel.papel)
    return {
        "Acao escolhida": papel,
        "Proximo_fechamento": valor_previsto
    }
