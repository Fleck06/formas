import cv2  # Importação do Opencv-Python
import numpy as np
import matplotlib.pyplot as plt

""" Classe que fará o carregamento e leitura da imagem"""


class FormDetector:
    def load_image(self, path_image):
        image = cv2.imread(path_image)
        if image is None:
            raise ValueError(f"Imagem não encontrada: {path_image}")
        return image

    def convert_to_grayscale(self, image):
        """ Transforma imagens coloridas em escala de cinza """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_blur(self, image):
        """ Aplica um desfoque para redizir os ruídos """
        return cv2.GaussianBlur(image, (5, 5), 0)

    def edge_detection(self, image):
        """ Detecta as bordas da imagem usando o método Canny """
        return cv2.Canny(image, 50, 150)

    def find_lines(self, imagem_bordas):
        """ Encontra os contornos da imagem """
        contornos, _ = cv2.findContours(
            imagem_bordas, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contornos

    def classify_forms(self, contornos, imagem_original):
        """ Classifica as formas geométricas. Classificaremos
        em 4 tipos: Triangulos, Quadrados, Círculos e outros"""
        formas = {
            'triangulos': 0,
            'quadrados': 0,
            'circulos': 0,
            'outros': 0
        }

        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            aproximacao = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)
            # Lógica de classificação
            if len(aproximacao) == 3:
                formas['triangulos'] += 1
                cv2.drawContours(imagem_original, [contorno], 0, (0, 255, 0), 2)
            elif len(aproximacao) == 4:
                formas['quadrados'] += 1
                cv2.drawContours(imagem_original, [contorno], 0, (255, 0, 0), 2)
            elif len(aproximacao) > 4:
                area = cv2.contourArea(contorno)
                perimetro_quadrado = perimetro * perimetro
                circularity = 4 * np.pi * area / perimetro_quadrado

                if circularity > 0.8:
                    formas['circulos'] += 1
                    cv2.drawContours(imagem_original, [contorno], 0, (0, 0, 255), 2)

                else:
                    formas['outros'] += 1
        return formas

    def view_results(self, imagem_original, formas):
        """ Visualizar os resultados """
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB))
        plt.title('Formas geométricas detectadas')
        plt.axis('off')
        legenda = f"""
Formas detectadas:
Triangulos: {formas['triangulos']}
Quadrados: {formas['quadrados']}
Circulos: {formas['circulos']}
Outros: {formas['outros']}
"""
        plt.text(imagem_original.shape[1],
                 imagem_original.shape[0],
                 legenda,
                 fontsize = 10,
                 verticalalignment = 'bottom',
                 horizontalalignment = 'right')

        plt.tight_layout()
        plt.show()

def main():
    caminho_imagem = 'formas.jpg'
    detector = FormDetector()

    try:
        imagem = detector.load_image(caminho_imagem)
        imagem_cinza = detector.convert_to_grayscale(imagem)
        imagem_blur = detector.apply_blur(imagem_cinza)
        imagem_bordas = detector.edge_detection(imagem_blur)
        contornos = detector.find_lines(imagem_bordas)
        formas = detector.classify_forms(contornos, imagem)
        detector.view_results(imagem, formas)
    except Exception as e:
        print(f'Erro no processamento da imagem: {e}')

if __name__ == "__main__":
        main()
