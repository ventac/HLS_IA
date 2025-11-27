import json

def convert_json_to_c_arrays(json_data, output_file):
    with open(output_file, 'w') as f:
        for layer_name, layer_data in json_data.items():
            f.write(f"// Layer: {layer_name}\n")
            
            # Handle kernels (weights)
            if "kernel" in layer_data:
                kernel = layer_data["kernel"]
                f.write(f"float {layer_name.upper()}_KERNEL[] = {{\n")
                for i, weight in enumerate(kernel):
                    f.write(f"    {weight:.8f},\n" if i < len(kernel) - 1 else f"    {weight:.8f}\n")
                f.write("};\n\n")
            
            # Handle biases
            if "bias" in layer_data:
                bias = layer_data["bias"]
                f.write(f"float {layer_name.upper()}_BIAS[] = {{\n")
                for i, b in enumerate(bias):
                    f.write(f"    {b:.8f},\n" if i < len(bias) - 1 else f"    {b:.8f}\n")
                f.write("};\n\n")
                
    print(f"Generated C code has been saved to {output_file}")

# Example: Load your JSON data from a file
json_file = '/media/se5-g2/Jarod SSD/Etudes/5e annee/Projet SE/HLS_IA/etc/poids.json'  # Path to your JSON file containing weights
output_file = '/media/se5-g2/Jarod SSD/Etudes/5e annee/Projet SE/HLS_IA/etc/poids.c'  # Output file with the C arrays

# Load the JSON data
with open(json_file, 'r') as f:
    json_data = json.load(f)

convert_json_to_c_arrays(json_data, output_file)
