import requests

# Substitua a URL pelo endpoint do seu servidor FastAPI
base_url = "http://localhost:8000"

# Exemplo de chamada para a rota de detecção
def detect_image(file_path):
    detect_url = f"{base_url}/detect"

    files = {'file': ('image.jpg', open(file_path, 'rb'))}
    data = {'img_size': 640, 'download_image': False}

    response = requests.post(detect_url, files=files, data=data)

    if response.status_code == 200:
        # Resultados em formato JSON
        json_results = response.json()
        print(json_results)
    else:
        print(f"Erro ao chamar a API. Código de status: {response.status_code}")
        print(response.text)

# Substitua 'path/para/sua/imagem.jpg' pelo caminho da sua imagem
detect_image('path/para/sua/imagem.jpg')
