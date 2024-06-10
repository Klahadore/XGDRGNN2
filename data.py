
from torch_geometric.data import HeteroData
import pickle
import os
from torch_geometric import transforms as T
from sklearn.decomposition import PCA
import json
import torch

from data_utils import *


# extract_zip(download_url(URL, 'data'), 'data')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FOLDER_PATH = "data/Gene_Disease_Network"

pickled_data_path = "data/pickled_data.pickle"

torch.manual_seed(42)



def build_dataset():
    gene_to_index = {}
    disease_to_index = {}
    chemical_to_index = {}
    phe_to_index = {}
    mutation_to_index = {}
    pathway_to_index = {}

    symptom_to_index = {}


    # Index is the vector corresponding to each specific node. Each node has an index and a tensor representing it.
    # keys are the nodes in the dataset who are defined in the json as a key
    # values are defined in the json dataset in the value part of the json. 
    build_index_map_of_keys("gene_disease.json", gene_to_index)
    build_index_map_of_keys("gene_mutation.json", gene_to_index)
    build_index_map_of_keys("gene_phe.json", gene_to_index)
    build_index_map_of_keys("gene_pathway.json", gene_to_index)
    build_index_map_of_keys("gene_chemical.json", gene_to_index)
    build_index_map_of_keys("gene_gene.json", gene_to_index)
    build_index_map_of_values("gene_gene.json", gene_to_index)
    build_index_map_of_values("disease_gene.json", gene_to_index)

    build_index_map_of_keys("disease_gene.json", disease_to_index)
    build_index_map_of_keys("disease_mutation.json", disease_to_index)
    build_index_map_of_keys("disease_phe.json", disease_to_index)
    build_index_map_of_keys("disease_pathway.json", disease_to_index)
    build_index_map_of_keys("disease_chemical.json", disease_to_index)
    build_index_map_of_keys("disease_disease.json", disease_to_index)
    build_index_map_of_values("disease_disease.json", disease_to_index)
    build_index_map_of_values("gene_disease.json", disease_to_index)

    build_index_map_of_values("disease_chemical.json", chemical_to_index)
    build_index_map_of_values("gene_chemical.json", chemical_to_index)

    build_index_map_of_values("disease_phe.json", phe_to_index)
    build_index_map_of_values("gene_phe.json", phe_to_index)

    build_index_map_of_values("disease_mutation.json", mutation_to_index)
    build_index_map_of_values("gene_mutation.json", mutation_to_index)

    build_index_map_of_values("disease_pathway.json", pathway_to_index)
    build_index_map_of_values("gene_pathway.json", pathway_to_index)

    build_index_map_of_values("disease_symptom.json", symptom_to_index)

    print("built index map")
    print(symptom_to_index)

    dataset = HeteroData()
    dataset['symptom'].x = create_node_embedding_tensor(symptom_to_index)
    print("I built Symptom embeddings")
    dataset['gene'].x = create_node_embedding_tensor(gene_to_index)
    dataset['disease'].x = create_node_embedding_tensor(disease_to_index)
    dataset['chemical'].x = create_node_embedding_tensor(chemical_to_index)
    dataset['phe'].x = create_node_embedding_tensor(phe_to_index)
    dataset['mutation'].x = create_node_embedding_tensor(mutation_to_index)
    dataset['pathway'].x = create_node_embedding_tensor(pathway_to_index)


    save_data(dataset, "hetero_dataset_with_node_embeddings.pkl")



    dataset['gene', 'gene_disease', 'disease'].edge_index = create_edge_indices(
        build_file_mapping("gene_disease.json"),
        gene_to_index, disease_to_index)
    dataset['gene', 'gene_chemical', 'chemical'].edge_index = create_edge_indices(
        build_file_mapping("gene_chemical.json"),
        gene_to_index, chemical_to_index)
    dataset['gene', 'gene_phe', 'phe'].edge_index = create_edge_indices(build_file_mapping("gene_phe.json"),
                                                                               gene_to_index, phe_to_index)
    dataset['gene', 'gene_mutation', 'mutation'].edge_index = create_edge_indices(
        build_file_mapping("gene_mutation.json"),
        gene_to_index, mutation_to_index)
    dataset['gene', 'gene_pathway', 'pathway'].edge_index = create_edge_indices(
        build_file_mapping("gene_pathway.json"),
        gene_to_index, pathway_to_index)
    dataset['gene', 'gene_gene', 'gene'].edge_index = create_edge_indices(build_file_mapping("gene_gene.json"),
                                                                                gene_to_index, gene_to_index)
    dataset['disease', 'disease_chemical', 'chemical'].edge_index = create_edge_indices(
        build_file_mapping("disease_chemical.json"),
        disease_to_index, chemical_to_index)
    dataset['disease', 'disease_phe', 'phe'].edge_index = create_edge_indices(
        build_file_mapping("disease_phe.json"),
        disease_to_index, phe_to_index)
    dataset['disease', 'disease_mutation', 'mutation'].edge_index = create_edge_indices(
        build_file_mapping("disease_mutation.json"),
        disease_to_index, mutation_to_index)
    dataset['disease', 'disease_pathway', 'pathway'].edge_index = create_edge_indices(
        build_file_mapping("disease_pathway.json"),
        disease_to_index, pathway_to_index)
    dataset['disease', 'disease_disease', 'disease'].edge_index = create_edge_indices(
        build_file_mapping("disease_disease.json"),
        disease_to_index, disease_to_index)

    dataset['disease', 'disease_symptom', 'symptom'].edge_index = create_edge_indices(build_file_mapping("disease_symptom.json"), disease_to_index, symptom_to_index)

    save_data(dataset, "hetero_dataset_with_node_embeddings_and_edge_index.pkl")

    # with open('data/new_connection_embedding.pkl', 'rb') as file:
    #     edge_dict = pickle.load(file)

    # Generate Edge Dict

    # dataset['gene', 'gene_disease', 'disease'].edge_attr = torch.tensor(edge_dict['gene_disease'])
    # dataset['gene', 'gene_chemical', 'chemical'].edge_attr = torch.tensor(edge_dict['gene_chemical'])
    # dataset['gene', 'gene_phe', 'phe'].edge_attr = torch.tensor(edge_dict['gene_phe'])
    # dataset['gene', 'gene_mutation', 'mutation'].edge_attr = torch.tensor(edge_dict['gene_mutation'])
    # dataset['gene', 'gene_pathway', 'pathway'].edge_attr = torch.tensor(edge_dict['gene_pathway'])
    # dataset['gene', 'gene_gene', 'gene'].edge_attr = torch.tensor(edge_dict['gene_gene'])
    # dataset['disease', 'disease_chemical', 'chemical'].edge_attr = torch.tensor(edge_dict['disease_chemical'])
    # dataset['disease', 'disease_phe', 'phe'].edge_attr = torch.tensor(edge_dict['disease_phe'])
    # dataset['disease', 'disease_mutation', 'mutation'].edge_attr = torch.tensor(edge_dict['disease_mutation'])
    # dataset['disease', 'disease_pathway', 'pathway'].edge_attr = torch.tensor(edge_dict['disease_pathway'])
    # dataset['disease', 'disease_disease', 'disease'].edge_attr = torch.tensor(edge_dict['disease_disease'])
    # dataset['disease', 'disease_symptom', 'symptom'].edge_attr = torch.tensor(edge_dict['disease_symptom'])

    #At this point the dataset has no edge_attr
    return dataset


# dataset = None
# if os.path.exists(pickled_data_path) and os.path.getsize(pickled_data_path) > 0:
#     try:
#         with open(pickled_data_path, 'rb') as file:
#             dataset = pickle.load(file)
#     except:
#         print("data set failed to load")

#     if dataset:
#         print("Loaded pickled data")

#     else:
#         print("Pickled data is empty")
#         dataset = build_dataset()
#         with open(pickled_data_path, 'wb') as file:
#             pickle.dump(dataset, file)
#         print("pickled dataset saved to file")
# else:
#     print("Pickled data file does not exist or is empty")
#     dataset = build_dataset()
#     with open(pickled_data_path, 'wb') as file:
#         pickle.dump(dataset, file)
#     print("Pickled data saved to file")
dataset = build_dataset()

dataset = T.ToDevice(device)(dataset)
dataset = T.ToUndirected()(dataset)
# dataset = T.NormalizeFeatures()(dataset)



transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    add_negative_train_samples=True,
    edge_types=("gene", "gene_disease", "disease",),
    rev_edge_types=("disease", "rev_gene_disease", "gene",)

)
train_dataset, val_dataset, test_dataset = transform(dataset)
print(dataset)
print(train_dataset)
del dataset
# del val_dataset
# del test_dataset
with open("temp_train.ppickle", 'wb') as file:
    pickle.dump(train_dataset, file)

with open("temp_val.pickle", 'wb') as file:
    pickle.dump(val_dataset, file)


with open("temp_test.pickle", 'wb') as file:
    pickle.dump(test_dataset, file)



# def build_edge_attr(het_dataset, edge_type, edge_type_mapping):
#     arr = []
#     for i in het_dataset.metadata()[1]:
#         print(i)
#         src, middle, dst = i
#         if "rev" in middle:
#             arr.append(het_dataset.edge_attr_dict[(dst, middle[4:], src)])
#         else:
#             arr.append(het_dataset.edge_attr_dict[i])

#     arr = torch.stack(arr)
#     print(arr.shape)
#     pca = PCA(n_components=22)

#     # Fit PCA on the data and transform the data
#     arr = torch.tensor(pca.fit_transform(arr.cpu().numpy()))
#     arr.to(device)
#     final_tensor = []
#     for i in edge_type.tolist():
#         src, middle, dst = edge_type_mapping[i]
#         if "rev" in middle:
#             index = het_dataset.metadata()[1].index((dst, middle[4:], src))
#             final_tensor.append(arr[index])
#         else:
#             index = het_dataset.metadata()[1].index((src, middle, dst))
#             final_tensor.append(arr[index])

#     return torch.stack(final_tensor)




def build_homo_dataset(hetero_dataset, name):
    print(hetero_dataset.metadata())
    new_dataset = hetero_dataset.to_homogeneous(dummy_values=True)


    # del hetero_dataset['gene'].x
    # del hetero_dataset['disease'].x
    # del hetero_dataset['phe'].x
    # del hetero_dataset['chemical'].x
    # del hetero_dataset['mutation'].x

    edge_type_embedding_mapping = generate_edge_attr_from_text(hetero_dataset.metadata())
    # new_dataset.edge_attr = build_edge_attr(hetero_dataset, new_dataset.edge_type, mapping)
    new_dataset.edge_attr = build_edge_attr_tensor(edge_type_embedding_mapping, new_dataset.edge_type)
   #  new_dataset.edge_attr = generate_new_edge_attr_tensor(mapping, new_dataset.edge_type, hetero_dataset)

    with open(f"data/{name}.pickle", 'wb') as file:
        pickle.dump(new_dataset, file)
    print("successfully built homogenous dataset and dumped pickled file at" + f"data/{name}.pickle")
    print(new_dataset)
    # return dataset




if not os.path.exists("data/train_dataset_metadata.pickle"):
    with open("data/train_dataset_metadata.pickle", 'wb') as file:
        pickle.dump(train_dataset.metadata(), file)
        print("dumped train_dataset.metadata()")


if not os.path.exists("data/new_train_dataset.pickle"):
    train_dataset = T.ToDevice(device)(train_dataset)
    build_homo_dataset(train_dataset, "new_train_dataset")
    print("built new_train_dataset")
    del train_dataset
# del train_dataset
# with open("data/new_train_dataset.pickle", "rb") as file:
#     new_train_dataset = pickle.load(file)
#     print("loaded new_train_dataset")
#
if not os.path.exists("data/new_test_dataset.pickle"):
    with open("temp_test.pickle", 'rb') as file:
        test_dataset = pickle.load(file)
    test_dataset = T.ToDevice(device)(test_dataset)
    build_homo_dataset(test_dataset, "new_test_dataset")
    print("built new_test_dataset")
    del test_dataset

if not os.path.exists("data/new_val_dataset.pickle"):
    with open("temp_val.pickle", 'rb') as file:
        val_dataset = pickle.load(file)
    
    val_dataset = T.ToDevice(device)(val_dataset)
    build_homo_dataset(val_dataset, "new_val_dataset")
    
    del val_dataset

    print("built new_val_dataset")

# with open("data/new_val_dataset.pickle", 'rb') as file:
#     new_val_dataset = pickle.load(file)
#     print("loaded new_val_dataset")

# with open("data/new_test_dataset.pickle", 'rb') as file:
#     new_test_dataset = pickle.load(file)
#     print("loaded new_test_dataset")




