import h5py
import numpy as np
import json

def read_h5_file(h5_filename):
    """
    Lit un fichier H5 et extrait les poids et les biais.
    
    :param h5_filename: Chemin vers le fichier H5.
    :return: Un dictionnaire avec les poids et les biais des couches.
    """
    weights = {}

    with h5py.File(h5_filename, 'r') as f:
        # Accède aux poids de chaque couche spécifiée
        for layer_name in f['/layers']:
            layer = f['/layers'][layer_name]
            if 'vars' in layer:
                # Extraction des poids et des biais dans '/layers/conv2d/vars'
                for var_name in layer['vars']:
                    if '0' in var_name:  # Si c'est un poids
                        weights[layer_name] = {
                            'kernel': np.array(layer['vars/0']),
                        }
                    elif '1' in var_name:  # Si c'est un biais
                        weights[layer_name]['bias'] = np.array(layer['vars/1'])
    
    return weights

def save_to_simple_text(weights, output_filename):
    """
    Sauvegarde les poids et les biais dans un fichier texte simple.
    
    :param weights: Dictionnaire des poids et des biais.
    :param output_filename: Chemin vers le fichier de sortie.
    """
    with open(output_filename, 'w') as f:
        for layer_name, layer_data in weights.items():
            f.write(f"Layer: {layer_name}\n")
            if 'kernel' in layer_data:
                kernel_values = layer_data['kernel'].flatten()
                kernel_values_str = ' '.join(map(str, kernel_values))
                f.write(f"  Kernel: {kernel_values_str}\n")
            if 'bias' in layer_data:
                bias_values = layer_data['bias'].flatten()
                bias_values_str = ' '.join(map(str, bias_values))
                f.write(f"  Bias: {bias_values_str}\n")
            f.write("\n")

def save_to_simple_json(weights, output_filename):
    """
    Sauvegarde les poids et les biais dans un fichier JSON simple.
    
    :param weights: Dictionnaire des poids et des biais.
    :param output_filename: Chemin vers le fichier de sortie.
    """
    simple_weights = {}

    for layer_name, layer_data in weights.items():
        simple_weights[layer_name] = {}
        if 'kernel' in layer_data:
            simple_weights[layer_name]['kernel'] = layer_data['kernel'].flatten().tolist()
        if 'bias' in layer_data:
            simple_weights[layer_name]['bias'] = layer_data['bias'].flatten().tolist()

    with open(output_filename, 'w') as f:
        json.dump(simple_weights, f, separators=(',', ':'))

def main():
    h5_filename = '/media/se5-g2/Jarod SSD/Etudes/5e annee/Projet SE/HLS_IA/etc/lenet_weights.weights.h5'  # Changez le chemin vers votre fichier H5
    output_text_filename = '/media/se5-g2/Jarod SSD/Etudes/5e annee/Projet SE/HLS_IA/etc/poids.txt'
    output_json_filename = '/media/se5-g2/Jarod SSD/Etudes/5e annee/Projet SE/HLS_IA/etc/poids.json'

    print(f"Lecture du fichier H5 : {h5_filename}")
    weights = read_h5_file(h5_filename)

    print(f"Enregistrement des poids dans un fichier texte simple : {output_text_filename}")
    save_to_simple_text(weights, output_text_filename)

    print(f"Enregistrement des poids dans un fichier JSON simple : {output_json_filename}")
    save_to_simple_json(weights, output_json_filename)

    print("Conversion terminée !")

if __name__ == "__main__":
    main()
