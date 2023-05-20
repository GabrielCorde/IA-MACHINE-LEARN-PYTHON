
# Passo 1: Entendimento do Desafio
# Passo 2: Entendimento da Área/Empresa


# Passo 3: Extração/Obtenção de Dados
import pandas as pd


tabela = pd.read_csv('barcos_ref.csv')
display(tabela)

# Passo 4: Ajuste de Dados (Tratamento/Limpeza)
print(tabela.info())

# Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)

#Preparação

# Separar a base de dados em X e Y

y = tabela['Preco']
x = tabela.drop("Preco", axis=1)

## separando os dados de treino e teste
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.3, random_state=1)

# Criação e treino da IA

# importar a inteligencia artificial
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# criar a inteligencia artificial

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treinar a inteligencia artificial

modelo_regressaolinear.fit(x_treino,y_treino)
modelo_arvoredecisao.fit(x_treino,y_treino)

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

from sklearn.metrics import r2_score

print(r2_score(y_teste,previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))

# Passo 7: Interpretação de Resultados

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar["Previsoes ARvore Decisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes Regressao Linear"] = previsao_regressaolinear

sns.lineplot(data=tabela_auxiliar)
plt.show()


tabela_nova = pd.read_csv("novos_barcos.csv")
display(tabela_nova)
previsao = modelo_arvoredecisao.predict(tabela_nova)
print(previsao)




# In[ ]:




