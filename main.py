import os
import cv2
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO ---
# Coloque aqui o caminho para a pasta principal onde estão as subpastas (rock, paper, scissors)
# Exemplo Windows: r'C:\Users\Voce\Downloads\rps-dataset'
# Exemplo Linux/Mac: '/home/voce/rps-dataset'
DIRETORIO_DATASET = '/home/jotapecds/UFRJ/Materias/cocada/trabalho_final/archive/rps-cv-images'

# Nomes das subpastas (verifique se no seu computador estão em inglês ou português)
CATEGORIAS = ['rock', 'paper', 'scissors'] 
# Se suas pastas chamam "pedra", "papel", "tesoura", mude a lista acima.

def visualizar_amostras():
    plt.figure(figsize=(15, 5))

    for indice, categoria in enumerate(CATEGORIAS):
        # Cria o caminho completo: dataset/rock, dataset/paper, etc.
        path = os.path.join(DIRETORIO_DATASET, categoria)
        
        # Verifica se a pasta existe para evitar erros
        if not os.path.exists(path):
            print(f"Atenção: A pasta '{path}' não foi encontrada.")
            continue
            
        # Pega o primeiro arquivo de imagem dentro da pasta
        for img_nome in os.listdir(path):
            if img_nome.endswith(('.png', '.jpg', '.jpeg')): # Garante que é imagem
                
                # Lê a imagem usando OpenCV
                img_array = cv2.imread(os.path.join(path, img_nome))
                
                # OpenCV carrega cores como BGR, mas o Matplotlib usa RGB. Vamos converter:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                
                # Plota a imagem
                plt.subplot(1, 3, indice + 1)
                plt.imshow(img_rgb)
                plt.title(f"Categoria: {categoria}\nDimensões: {img_array.shape}")
                plt.axis('off') # Remove os eixos com números
                
                break # Para o loop interno após pegar a primeira imagem de cada pasta
    
    plt.tight_layout()
    plt.show()

# Executa a função
visualizar_amostras()