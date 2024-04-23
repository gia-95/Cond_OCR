import os

def inserisci_in_CF_file(word, img_filename, label_cf_filename, directory) :
    tupla_CF_to_insert = f'(\'{directory}/{img_filename}\', None, \'{word}\'), '
    with open(label_cf_filename, 'a') as f:
        f.write(tupla_CF_to_insert)
    print("  *** scrittura codice fiscale")



def get_labels_from_file(lables_file_path) :
    tuple = []
    file = open(lables_file_path,'r')
    file_content = file.read()[1:]
    array_tuple = file_content.split('), (')
    
    if (len(array_tuple) == 1) :
        return []

    for elem in array_tuple :
        temp_array = elem.split('\'')
        tupla = (temp_array[1], None, temp_array[3])
        tuple.append(tupla)

    return tuple


def get_larger_index_dataset(dir) :
    id_last_image = 0 
    for file in os.listdir(dir) :
        if (file == '.DS_Store') : continue
        id_file_corrente =  int(file.split('_')[1].split('.')[0])
        if (id_file_corrente >= id_last_image) : id_last_image = id_file_corrente
    return id_last_image

