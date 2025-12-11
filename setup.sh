#!/bin/bash
# setup.sh - Script de configuração do projeto

echo "Configurando ambiente para o projeto Rock-Paper-Scissors..."

# 1. Instala dependências do sistema
echo "Instalando dependências do sistema..."
sudo apt update
sudo apt install -y python3-venv python3-pip python3-full

# 2. Cria ambiente virtual
echo "Criando ambiente virtual..."
python3 -m venv venv

# 3. Ativa ambiente virtual
echo "Ativando ambiente virtual..."
source venv/bin/activate

# 4. Instala dependências Python
echo "Instalando pacotes Python..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verifica estrutura do projeto
echo "Verificando estrutura do projeto..."
if [ ! -d "data" ]; then
    echo "AVISO: Pasta 'data' não encontrada!"
    echo "Baixe o dataset de: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors"
    echo "Extraia para a pasta 'data/' mantendo as subpastas rock, paper, scissors"
fi

echo "Configuração completa!"
echo ""
echo "Para ativar o ambiente virtual: source venv/bin/activate"
echo "Para executar o projeto: python3 main.py"