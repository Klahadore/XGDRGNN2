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










print(resolve_id_to_name("D003316"))
print(input_disease_output_symptoms("D00764", "disease_disease.json"))