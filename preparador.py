import os
import shutil


def copiarAnalysis():
    print("Copiando arquivos")
    shutil.copytree("./datasets/originais/analysis",
                    "./datasets/preparados/analysis")


def prepararAnalysis():
    # Remover arquivo defeituoso
    print("Removendo arquivo defeituoso")
    path = "./datasets/preparados/analysis"
    arquivoRemover = "/neutrophil/.DS_169665.jpg"
    os.remove(path + arquivoRemover)

    print("Preparação base finalizada")
    # Renomear pastas


def copiarBCCD():
    print("Copiando arquivos")
    shutil.copytree("./datasets/originais/bccd/BCCD",
                    "./datasets/preparados/bccd")


def prepararBCCD():
    # Remover pasta inútil
    print("Removendo pasta")
    path = "./datasets/preparados/bccd"
    arquivoRemover = "/ImageSets"
    shutil.rmtree(path + arquivoRemover)

    print("Preparação base finalizada")
    # Renomear pastas
