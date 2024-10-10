#!/usr/bin/env python
# coding: utf-8

# # Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns 

# ### Contexto
# 
# No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# 
# Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
# 
# Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)
# 
# Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.
# 
# ### Nosso objetivo
# 
# Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
# 
# Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.
# 
# ### O que temos disponível, inspirações e créditos
# 
# As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# 
# Elas estão disponíveis para download abaixo da aula (se você puxar os dados direto do Kaggle pode ser que encontre resultados diferentes dos meus, afinal as bases de dados podem ter sido atualizadas).
# 
# Caso queira uma outra solução, podemos olhar como referência a solução do usuário Allan Bruno do kaggle no Notebook: https://www.kaggle.com/allanbruno/helping-regular-people-price-listings-on-airbnb
# 
# Você vai perceber semelhanças entre a solução que vamos desenvolver aqui e a dele, mas também algumas diferenças significativas no processo de construção do projeto.
# 
# - As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
# - Os preços são dados em reais (R$)
# - Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados
# 
# ### Expectativas Iniciais
# 
# - Acredito que a sazonalidade pode ser um fator importante, visto que meses como dezembro costumam ser bem caros no RJ
# - A localização do imóvel deve fazer muita diferença no preço, já que no Rio de Janeiro a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos)
# - Adicionais/Comodidades podem ter um impacto significativo, visto que temos muitos prédios e casas antigos no Rio de Janeiro
# 
# Vamos descobrir o quanto esses fatores impactam e se temos outros fatores não tão intuitivos que são extremamente importantes.

# ### Importar Bibliotecas e Bases de Dados

# In[1]:


import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import folium
from IPython.display import display
import branca


# ### Consolidar Base de Dados

# In[2]:


meses = {'jan': 1, 'fev': 2, 'mar': 3,'abr': 4,'mai': 5,'jun': 6,'jul': 7,'ago': 8,'set': 9,'out': 10,'nov': 11,'dez': 12}


caminho_bases = pathlib.Path('dataset')
base_airbnb = pd.DataFrame()
for arquivo in caminho_bases.iterdir():
    
    nome_mes = arquivo.name[:3]
    #num mes
    mes = meses[nome_mes]
    ano = arquivo.name[-8:]
    ano = int(ano.replace(".csv",""))
    
    
    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = pd.concat([base_airbnb,df],ignore_index=True)
display(base_airbnb)


# - Como temos muitas colunas, nosso modelo pode acabar ficando lento.
# - Além disso, fazendo uma análise rápida foi possibilitado a identificação de que várias colunas não são necessárias para o modelo de previsão, por isso, serão exclusas da base.
# - Tipos de colunas que serão excluídas:
#     1. Id's links e informações não relevantes para o modelo.
#     2. Colunas repetidas ou extremamente parecidas com outra (dão a mesma info para o modelo Ex: Data x Ano/Mês
#     3. Colunas preenchidas com texto livre, não será preciso
#     4. Colunas em que todos ou quase todos os valores são iguais
# - Para isso, será criado um arquivo excel com os 1.000 primeiros registros para que seja feita uma análise qualitativa.
# - Note que as mudanças foram no sentido de deixar somente colunas que impactam no preço da diária.

# ### Se tivermos muitas colunas, já vamos identificar quais colunas podemos excluir
# 
# - Gerar os primeiros mil arquivos em csv para identificar as colunas que serão excluidas

# In[3]:


print(list(base_airbnb.columns))

base_airbnb.head(1000).to_csv('primeiros_resgistros.csv', sep=';')


# ###  Depois da análise qualitativa  das colunas, levando em conta os critérios acima aplicados, as colunuas ficaram assim:

# In[4]:


colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']
base_airbnb = base_airbnb.loc[:,colunas]
display(base_airbnb)


# ### Tratar Valores Faltando
# 
# - Visualizando os dados, constatou-se que existia uma grande disparidade em dados faltantes. As colunas com mais de 300.000 valores NaN foram excluídas da análise.
# 
# - Para as outras colunas, foram excluídas apenas as linhas NaN (haja vista que temos mais de 900 k de linhas)
# 

# In[5]:


for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna,axis=1)


base_airbnb = base_airbnb.dropna()
print(base_airbnb.shape)
print(base_airbnb.isnull().sum())


# ### Verificar Tipos de Dados em cada coluna

# In[6]:


print(base_airbnb.dtypes)
print('-' * 75)
print(base_airbnb.iloc[0])



# - Como o preço e preço extra por pessoa estão reconhecidos como Objeto, alteramos as colunas para aparecerem como float

# In[7]:


#price
try:
    base_airbnb['price'] = base_airbnb['price'].str.replace('$', '').str.replace(',', '')
    base_airbnb['price'] = base_airbnb['price'].astype(np.float64, copy=False)
    
    base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '').str.replace(',', '')
    base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float64, copy=False)
except:
    print('As variáveis já foram convertidas em Float64!')

print(base_airbnb.dtypes)


# ### Análise Exploratória e Tratar Outliers
# 
# - Vamos basicamente olhar feature por feature para:
#     1. Ver a correlação entre as features e decidir se manteremos todas as features que temos.
#     2. Excluir outliers (usaremos como regra, valores abaixo de Q1 - 1.5xAmplitude e valores acima de Q3 + 1.5x Amplitude). Amplitude = Q3 - Q1
#     3. Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma delas não vai nos ajudar e se devemos excluir
#     
# - Vamos começar pelas colunas de preço (resultado final que queremos) e de extra_people (também valor monetário). Esses são os valores numéricos contínuos.
# 
# - Depois vamos analisar as colunas de valores numéricos discretos (accomodates, bedrooms, guests_included, etc.)
# 
# - Por fim, vamos avaliar as colunas de texto e definir quais categorias fazem sentido mantermos ou não.
# 
# MAS CUIDADO: não saia excluindo direto outliers, pense exatamente no que você está fazendo. Se não tem um motivo claro para remover o outlier, talvez não seja necessário e pode ser prejudicial para a generalização. Então tem que ter uma balança ai. Claro que você sempre pode testar e ver qual dá o melhor resultado, mas fazer isso para todas as features vai dar muito trabalho.
# 
# Ex de análise: Se o objetivo é ajudar a precificar um imóvel que você está querendo disponibilizar, excluir outliers em host_listings_count pode fazer sentido. Agora, se você é uma empresa com uma série de propriedades e quer comparar com outras empresas do tipo também e se posicionar dessa forma, talvez excluir quem tem acima de 6 propriedades tire isso do seu modelo. Pense sempre no seu objetivo

# In[8]:


# Criar uma tabela só com valores numéricos
numericos = base_airbnb.select_dtypes(include=['float64', 'int64'])

# Calcular a matriz de correlação
correlation_matrix = numericos.corr()



# Criar o colormap personalizado
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", [
    "#e0f3f9",  # Azul muito claro, quase branco
    "#b0d4ea",  # Azul claro
    "#7aaed4",  # Azul médio
    "#4f8ab3",  # Azul suave
    "#266890",  # Azul escuro
    "#1E4678",  # Azul principal
    "#163659",  # Azul escuro, tom complementar
    "#122b46",  # Azul muito escuro
    "#0f2034",  # Azul quase preto
    "#001827"   # Tom mais profundo, próximo ao preto
])

plt.figure(figsize=(20, 10))
plt.title("Mapa de calor das correlações")

# Criar o mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap=custom_cmap, fmt=".2f", cbar=True)

plt.show()


# ### Definição de Funções para Análise de Outliers
# -Vamos definir alguma funções para ajudar na análise de outliers das colunas

# In[9]:


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude


# In[10]:


def diagrama_caixa(coluna):
    fig,(ax1, ax2) = plt.subplots(1,2)
    
    #tamanho dos graficos
    fig.set_size_inches(15,5)
    
    
    #grafico 1 com limite automatico
    sns.boxplot(x=coluna,ax = ax1)
    
    #limite do segundo boxplot
    ax2.set_xlim(limites(coluna))
    
    sns.boxplot(x=coluna,ax = ax2)

def histograma(coluna):
    #configuração de tamanho para 1 grafico
    plt.figure(figsize=(15,5))
    sns.histplot(coluna, kde=True) 
    
def excluir_outliers(df,nome_coluna):
    qtde_linhas = df.shape[0]
    lim_if, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_if) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas

def grafico_barra(coluna):
    plt.figure(figsize=(15,5))
    ax = sns.barplot(x=coluna.value_counts().index,y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# In[11]:


print(limites(base_airbnb['price']))
base_airbnb['price'].describe()


# ### PRICE
# 

# In[12]:


diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])


# - Como o modelo é realizado para apartamentos comuns, entendo que as diárias acima do limite superior serão apenas para
# apartamentos de altíssimo padrão, que não é nosso caso, por isso podemos exclui-los.

# In[13]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'price')
print('{} linhas removidas'.format(linhas_removidas))


# In[14]:


histograma(base_airbnb['price'])


# ### Extra people

# In[15]:


diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])


# In[16]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'extra_people')
print('{} linhas removidas'.format(linhas_removidas))


# In[17]:


histograma(base_airbnb['extra_people'])


# ### host_listings_count   

# In[18]:


diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])


# - Exclui os outliers, porque para o objetivo do projeto hosts com mais de 6 imoveis no airbnb não são o público alvo.

# In[19]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'host_listings_count')
print('{} linhas removidas'.format(linhas_removidas))


# ### accommodates

# In[20]:


diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])


# In[21]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'accommodates')
print('{} linhas removidas'.format(linhas_removidas))


# ### bathrooms 

# In[22]:


diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15,5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index,y=base_airbnb['bathrooms'].value_counts())


# In[23]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'bathrooms')
print('{} linhas removidas'.format(linhas_removidas))


# ### bedrooms 

# In[24]:


diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])


# In[25]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'bedrooms')
print('{} linhas removidas'.format(linhas_removidas))


# ### Beds

# In[26]:


diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])


# In[27]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'beds')
print('{} linhas removidas'.format(linhas_removidas))


# ### guests_included  

# In[28]:


#diagrama_caixa(base_airbnb['guests_included'])
#grafico_barra(base_airbnb['guests_included'])
print(limites(base_airbnb['guests_included']))


# - Vamos remover essa feature da análise, há indícios de que os hosts do airbnb adotam como padrão informar 1 ou não informar guest included. Assim pode levar o modelo a considerar essa feature, como explicado ela não é essencial para a definição de preço.

# In[29]:


try:
    base_airbnb = base_airbnb.drop('guests_included',axis=1)
except:
    print('guests_included já excluída')
base_airbnb.shape


# ### minimum_nights

# In[30]:


diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])


# In[31]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'minimum_nights')
print('{} linhas removidas'.format(linhas_removidas))


# ### maximum_nights

# In[32]:


diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])


# - Vamos remover essa feature da análise, há indícios de que os hosts do airbnb adotam como padrão informar 0 ou um numero muito acima como maximo de noites. 
# Assim pode levar o modelo a considerar essa feature,como explicado ela não é essencial para a definição de preço.

# In[33]:


try:
    base_airbnb = base_airbnb.drop('maximum_nights',axis=1)
except:
    print('maximum_nights já excluída')
base_airbnb.shape


# ### number_of_reviews                  
# 

# In[34]:


diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])


# In[35]:


try:
    base_airbnb = base_airbnb.drop('number_of_reviews',axis=1)
except:
    print('number_of_reviews já excluída')
base_airbnb.shape


# ### Tratamento de colunas de texto

# ### property_type 

# In[36]:


print(base_airbnb['property_type'].value_counts())
print('-'*254)
plt.figure(figsize=(15,5))
sns.countplot(x='property_type',data=base_airbnb)
plt.xticks(rotation=90)
plt.show()


# In[37]:


tipos_casa = base_airbnb['property_type'].value_counts()

outras = []

for tipo in tipos_casa.index:
    if tipos_casa[tipo] < 2000:
        outras.append(tipo)
print(outras)
        
    
for tipo in outras:
    base_airbnb.loc[base_airbnb['property_type'] ==tipo,'property_type'] = 'Outros'
print(base_airbnb['property_type'].value_counts())
print('-'*254)
plt.figure(figsize=(15,5))
sns.countplot(x='property_type',data=base_airbnb)
plt.xticks(rotation=90)
plt.show()


# ###  room_type

# In[38]:


print(base_airbnb['room_type'].value_counts())
print('-'*254)
plt.figure(figsize=(15,5))
sns.countplot(x='room_type',data=base_airbnb)
plt.xticks(rotation=90)
plt.show()


# - Em tipo de quarto, não precisamos fazer nada, ele já parece estar bem distribuído

# ### bed_type

# In[39]:


print(base_airbnb['bed_type'].value_counts())
print('-'*254)
plt.figure(figsize=(15,5))
sns.countplot(x='bed_type',data=base_airbnb)
plt.xticks(rotation=90)
plt.show()


# In[40]:


#Agrupando  tipos de cama diferentes de camas reais
tipos_cama = base_airbnb['bed_type'].value_counts()

outras_camas = []

for tipo in tipos_cama.index:
    if tipos_cama[tipo] < 10000:
        outras_camas.append(tipo)
print(outras_camas)
        
    
for tipo in outras_camas:
    base_airbnb.loc[base_airbnb['bed_type'] ==tipo,'bed_type'] = 'Outros'
print(base_airbnb['bed_type'].value_counts())
print('-'*254)
plt.figure(figsize=(15,5))
sns.countplot(x='bed_type',data=base_airbnb)
plt.xticks(rotation=90)
plt.show()


# ### cancellation_policy 

# In[41]:


print(base_airbnb['cancellation_policy'].value_counts())
print('-'*254)
plt.figure(figsize=(15,5))
sns.countplot(x='cancellation_policy',data=base_airbnb)
plt.xticks(rotation=90)
plt.show()

#Agrupando as politicas estritas
tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
outras_politicas = []

for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 10000:
        outras_politicas.append(tipo)
print(outras_politicas)
        
    
for tipo in outras_politicas:
    base_airbnb.loc[base_airbnb['cancellation_policy'] ==tipo,'cancellation_policy'] = 'strict'
print(base_airbnb['cancellation_policy'].value_counts())
print('-'*254)
plt.figure(figsize=(15,5))
sns.countplot(x='cancellation_policy',data=base_airbnb)
plt.xticks(rotation=90)
plt.title("Políticas de Cancelamento")
plt.show()


# ### amenities
# 
# 
# + Como temos uma diversidade muito grande de amenities e, às vezes, os mesmos amenities podem ser escritos de diferentes formas, vamos avaliar a quantidade de amenities como o parâmetro para o nosso modelo.
# - A lógica que talvez possa ser usada pelos hosters é quanto mais amenities maior o preço.

# In[42]:


print(base_airbnb['amenities'].iloc[3].split(','))
print(len((base_airbnb['amenities'].iloc[3].split(','))))

#Criar a coluna quantidade de amenities por imovel
base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)


# In[43]:


#Excluir a coluna amenities.
try:
    base_airbnb = base_airbnb.drop('amenities',axis=1)
except:
    print('Coluna amenities já excluída!')
#visualizar o novo tamanho da base de dados
base_airbnb.shape


# In[44]:


diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])


# In[45]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb,'n_amenities')
print('{} linhas removidas'.format(linhas_removidas))
diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])


# ### Visualização de Mapa das Propriedades

# In[46]:


amostra = base_airbnb.sample(n=50000)
# Criar o mapa centrado em uma coordenada
centro_mapa = [amostra.latitude.mean(), amostra.longitude.mean()]
mapa = folium.Map(location=centro_mapa, zoom_start=10)

# Função para definir a cor com base no preço
def get_color(price):
    if price <= 200:
        return 'blue'
    elif 201 <= price <= 400:
        return 'lightgreen'
    elif 401 <= price <= 800:
        return 'darkgreen'
    elif 801 <= price <= 1000:
        return 'yellow'
    elif 1001 <= price <= 1300:
        return 'red'
    else:
        return 'Pink'  

# Adicionar pontos de calor no mapa com as cores personalizadas
for _, row in amostra.iterrows():
    folium.CircleMarker(location=[row['latitude'], row['longitude']],
                        radius=5, 
                        color=get_color(row['price']),  # Cor baseada no preço
                        fill=True, fill_color=get_color(row['price']),
                        fill_opacity=0.7).add_to(mapa)

# Salvar o mapa como arquivo HTML
mapa.save("mapa_airbnb.html")


mapa


# ### Encoding
# 
# Precisamor Ajustar as features para facilitar o trabalho do modelo futuro (features de categoria, true e false, etc.)
# 
# - Features de Valores True ou False, vamos substituir True por 1 e False por 0.
# - Features de Categoria (features em que os valores da coluna são textos) vamos utilizar o método de encoding de variáveis dummies

# In[47]:


colunas_tf = ['host_is_superhost', 'instant_bookable','is_business_travel_ready']

base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='t',coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='f',coluna] = 0





# In[48]:


colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categorias)
display(base_airbnb_cod.head())


# ### Modelo de Previsão
# 
# - Métricas de Avaliação
# 
# Vamos usar aqui o R² que vai nos dizer o quão bem o nosso modelo consegue explicar o preço. Isso seria um ótimo parâmetro para ver o quão bom é nosso modelo <br>
# -> Quanto mais próximo de 100%, melhor
# 
# Vou calcular também o Erro Quadrático Médio, que vai mostrar para gente o quanto o nosso modelo está errando. <br>
# -> Quanto menor for o erro, melhor

# In[49]:


def avaliar_modelo(nome_modelo,y_teste, previsao):
    r2 = r2_score(y_teste,previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR²:{r2:.2%}\nRSME:{RSME:.2f}'


# - Escolha dos Modelos a Serem Testados
#     1. RandomForest
#     2. LinearRegression
#     3. Extra Tree
#     
# Esses são alguns dos modelos que existem para fazer previsão de valores numéricos (o que chamamos de regressão). Estamos querendo calcular o preço, portanto, queremos prever um valor numérico.
# 
# Assim, escolhemos esses 3 modelos. Existem dezenas, ou até centenas de modelos diferentes. A medida com que você for aprendendo mais e mais sobre Ciência de Dados, você vai aprender sempre novos modelos e entendendo aos poucos qual o melhor modelo para usar em cada situação.
# 
# Mas na dúvida, esses 3 modelos que usamos aqui são muito bons para muitos problemas de Regressão.

# In[50]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price',axis=1)


# - Separa os dados em treino e teste + Treino do Modelo
# 
# Essa etapa é crucial. As Inteligências Artificiais aprendendo com o que chamamos de treino.
# 
# Basicamente o que a gente faz é: a gente separa as informações em treino e teste, ex: 10% da base de dados vai servir para teste e 90% para treino (normalmente treino é maior mesmo)
# 
# Aí, damos para o modelo os dados de treino, ele vai olhar aqueles dados e aprender a prever os preços.
# 
# Depois que ele aprende, você faz um teste com ele, com os dados de teste, para ver se ela está bom ou não. Analisando os dados de teste você descobre o melhor modelo

# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(X_train, y_train)
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo,y_test, previsao))


# ### Análise do Melhor Modelo
# - Modelo Escolhido como Melhor Modelo: ExtraTressRegressor
# 
#     Esse foi o modelo com maior valor de R² e ao mesmo tempo o menor valor de RSME. Como não tivemos uma grande diferença de velocidade de treino e de previsão desse modelo com o modelo de RandomForest (que teve resultados próximos de R² e RSME), vamos escolher o Modelo ExtraTrees.
#     
#     O modelo de regressão linear não obteve um resultado satisfatório, com valores de R² e RSME muito piores do que os outros 2 modelos.
#     
# - Resultados das Métricas de Avaliaçõ no Modelo Vencedor:<br>
# Modelo ExtraTrees:<br>
# R²:97.49%<br>
# RSME:42.01

# In[52]:


#print(modelo_et.feature_importances_)
#print(X_train.columns)

importancia_features = pd.DataFrame(modelo_et.feature_importances_,X_train.columns)
importancia_features = importancia_features.sort_values(by=0,ascending=False)
display(importancia_features)

plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index,y=importancia_features[0])
ax.tick_params(axis='x', rotation=90)


# ### Ajustes Finais no Modelo
# 
# - is_business_travel ready não parece ter muito impacto no nosso modelo. Por isso, para chegar em um modelo mais simples, vamos excluir essa feature e testar o modelo sem ela.

# In[53]:


base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready',axis=1)

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=10)


#treinar
modelo_et.fit(X_train, y_train)
#testar
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees',y_test, previsao))


# In[54]:


base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:    
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=10)


#treinar
modelo_et.fit(X_train, y_train)
#testar
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees',y_test, previsao))


# In[55]:


print(previsao)


# # Deploy do Projeto
# 
# - Passo 1 -> Criar arquivo do Modelo (joblib)<br>
# - Passo 2 -> Escolher a forma de deploy:
#     - Arquivo Executável + Tkinter
#     - Deploy em Microsite (Flask)
#     - Deploy apenas para uso direto Streamlit
# - Passo 3 -> Outro arquivo Python (pode ser Jupyter ou PyCharm)
# - Passo 4 -> Importar streamlit e criar código
# - Passo 5 -> Atribuir ao botão o carregamento do modelo
# - Passo 6 -> Deploy feito

# In[56]:


X['price'] = y
X.to_csv('dados.csv')


# In[57]:


import joblib
joblib.dump(modelo_et, 'modelo.joblib')

