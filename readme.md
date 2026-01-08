Projeto: Análise de Tempo de Ecrã vs. Bem-Estar Mental

Este repositório contém o código para o Projeto 1 da UC de Aprendizagem Automática em Sistemas Empresariais (AASE). O objetivo é explorar a relação entre o tempo de ecrã e o bem-estar mental, seguindo a metodologia CRISP-DM, com base no dataset "Screen Time vs Mental Wellness".

Pré-requisitos

Antes de começar, certifique-se de que tem o Python (versão 3.8 ou superior) instalado no seu sistema. A instalação do Python geralmente inclui o gestor de pacotes pip.

Instalação e Configuração

Siga estes passos para configurar o ambiente de desenvolvimento local e instalar as dependências necessárias.

1. Criar o Ambiente Virtual (venv)

recomenda-se a utilização de um ambiente virtual (venv) para isolar as dependências deste projeto.

Abra um terminal na raiz deste projeto e execute o seguinte comando:

No Windows:

`python -m venv .venv`


No macOS ou Linux:

`python3 -m venv .venv`


Isto irá criar uma pasta chamada venv no diretório do seu projeto.

2. Ativar o Ambiente Virtual

Antes de instalar qualquer pacote, precisa de "ativar" o ambiente que acabou de criar:

No Windows (Command Prompt):

`.\\venv\\Scripts\\activate`


No Windows (PowerShell):

`.\\venv\\Scripts\\Activate.ps1`


No macOS ou Linux:

`source .venv/bin/activate`


Após a ativação, deverá ver (venv) no início da linha do seu terminal.

3. Instalar as Dependências

Com o ambiente virtual ativo, instale todas as bibliotecas necessárias (como pandas, seaborn, matplotlib, jupyter, etc.) usando o ficheiro requirements.txt:

`pip install -r requirements.txt`


O pip irá descarregar e instalar automaticamente todas as dependências listadas no ficheiro.

Como Executar o Código

Depois da instalação estar concluída, pode executar o código do projeto que se encontra na pasta `notebooks` ou `src`.

# Certifique-se que o seu venv está ativo
`jupyter lab`

Este comando irá abrir automaticamente uma nova janela no seu navegador de internet.

No navegador, clique na pasta `notebooks/`.

Clique no ficheiro `1. Data Understanding.ipynb` para o abrir.
Pode agora executar cada célula de código dentro do notebook.

## Estrutura do Projeto (CRISP-DM)

- `data/`: Armazenamento de dados.
- `docs/`: Documentação do projeto (ex: ProjectPlan.pdf).
- `notebooks/`: Jupyter notebooks para exploração e modelagem (Data Understanding, Preparation, Modeling).
- `src/`: Código fonte da aplicação.
  - `etl/`: Pipeline de extração, transformação e carga.
  - `dashboard/`: Aplicação de dashboard.
  - `utils/`: Scripts utilitários.
- `models/`: Modelos serializados (.pkl).
- `.venv/`: Ambiente virtual.
- `docker-compose.yml`: Orquestração de containers.


4. Desativar o Ambiente Virtual

Quando terminar de trabalhar no projeto, pode desativar o ambiente virtual simplesmente executando no terminal:

`deactivate`

## Verificação do Data Warehouse (Star Schema)

Para verificar se a migração para o Star Schema foi bem-sucedida, você pode conectar-se ao banco de dados e executar as seguintes consultas:

1.  **Aceder ao container do banco de dados:**
    ```bash
    docker exec -it db_warehouse psql -U user -d mental_health
    ```

2.  **Listar as tabelas (deve ver `fact_mental_health`, `dim_user`, `dim_occupation`, `dim_work_mode`):**
    ```sql
    \dt
    ```

3.  **Verificar contagem de registos na tabela de fatos:**
    ```sql
    SELECT COUNT(*) FROM fact_mental_health;
    ```

4.  **Exemplo de consulta analítica (Média de Stress por Ocupação):**
    ```sql
    SELECT 
        o.occupation_name,
        AVG(f.stress_level_0_10) as avg_stress
    FROM fact_mental_health f
    JOIN dim_occupation o ON f.occupation_id = o.occupation_id
    GROUP BY o.occupation_name
    ORDER BY avg_stress DESC;
    ```
