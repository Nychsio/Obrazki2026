from datasets import load_dataset

def get_openfake_stream():
    # Streaming pozwala na iterowanie po danych bez pobierania całego zbioru (np. 100GB+)
    dataset = load_dataset("ComplexDataLab/OpenFake", streaming=True)
    
    # Przykład pobrania pierwszej próbki
    example = next(iter(dataset['train']))
    return dataset

# To pokaże rekruterowi, że rozumiesz optymalizację pipeline'u danych.