import requests



def resolve_id_to_name(id) -> str:
	url = f"https://id.nlm.nih.gov/mesh/{id}.json"
	
	with requests.get(url) as response:
		data = response.json()
	#	print(data)

	try:
		search = data["preferredConcept"]
	except:
		print("Resolve your own name: ", id)
		return(input())

	url = search + ".json"

	with requests.get(url) as response:
		data = response.json()

	name = data["label"]["@value"].lower()

	return name

# Symptoms are outputted as id's. resolve_id_to_name is used to turn
# ids into names
#  Call this function for every file. 
def input_disease_output_symptoms(id, filename):
	data_dictionary = {}

	with open(f"data/Gene_Disease_Network/{filename}", "r") as file:
		lines = file.readlines()[1:]
	for line in lines:
		columns = line.strip().split('\t')
		key = columns[5]
		value = columns[6]

		if key in data_dictionary:
			if value not in data_dictionary[key]:
				data_dictionary[key].append(value)
		else:
			data_dictionary[key] = [value]

	return data_dictionary


# test id to name
def get_embedding(text):
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return torch.tensor(model.encode(input(f"didnt find embedding, inpput definition, {id}")))


def get_text(text):
  response = ollama.chat(model="llama3", messages=[
    {
      'role': 'user',
      'content': f"Provide a comprehensive and detailed definition of the medical term '{text}'. Focus solely on explaining what the term means."
    },
  ])
  return response['message']['content']


def build_file_mapping(filename):
    with open(FOLDER_PATH + "/" + filename) as file:
        json_data = json.load(file)
    return json_data


def add_to_map_index(mapping, key, value):
    if key in mapping:
        pass
    else:
        mapping[key] = value


def build_index_map_of_keys(filename, index_map):
    json_data = build_file_mapping(filename)
    for key in json_data:
        add_to_map_index(index_map, key, len(index_map))


def build_index_map_of_values(filename, index_map):
    json_data = build_file_mapping(filename)
    for key in json_data:
        for value in json_data[key]:
            add_to_map_index(index_map, value, len(index_map))


"""
Builds node to index maps for each type of node
"""


def create_node_embedding_tensor(index_map):
    tensor_list = []
    for key in index_map:
        tensor_list.append(embedding_function(key))
    return torch.stack(tensor_list)


def create_edge_indices(file_mapping, index_map_1, index_map_2):
    # Calculate the total number of elements to preallocate
    total_elements = sum(len(values) for values in file_mapping.values())

    # Preallocate the tensor
    edge_indices = torch.empty(2, total_elements, dtype=torch.int64)

    # Fill the tensor
    current_index = 0
    for key in file_mapping:
        for value in file_mapping[key]:
            edge_indices[:, current_index] = torch.tensor([index_map_1[key], index_map_2[value]], dtype=torch.int64)
            current_index += 1

    return edge_indices

def generate_edge_type_map(metadata):
    mapping = {}
    counter = 0
    for i in metadata[1]:
        if i not in mapping.keys():
            mapping[i] = counter
            counter += 1
    inv_map = {v: k for k, v in mapping.items()}
    return inv_map



def generate_new_edge_attr_tensor(edge_type_mapping, edge_type, het_dataset):
    # Pre-determine the size of the tensor
    num_edges = len(edge_type)
    # Assuming each edge attribute is of the same size, get the size of the first attribute
   #  first_attr_size = het_dataset.edge_attr_dict[next(iter(het_dataset.edge_attr_dict))].size()

    # Preallocate tensor space
    edge_attrs = torch.empty((num_edges, 64))

    for idx, i in enumerate(edge_type.tolist()):
        src, middle, dst = edge_type_mapping[i]
        if "rev" in middle:
            edge_attrs[idx] = het_dataset.edge_attr_dict[(dst, middle[4:], src)]
        else:
            edge_attrs[idx] = het_dataset.edge_attr_dict[edge_type_mapping[i]]

    return edge_attrs










print(resolve_id_to_name("D003316"))
print(input_disease_output_symptoms("D00764", "disease_disease.json"))