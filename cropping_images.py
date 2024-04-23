import get_models
import os
import keras_ocr
import cv2
import utile
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

#### Il processo prende le immagini intere dalla cartella 'dir_documenti',
#### ritaglia le parole, e le salva nella cartella 'dir_temp_images'
#### Il processo dopo (labeling.py) prende le parole dalla cartella 'dir_temp_images',
#### e dopo aver settato la label, le salva nella cartella definitiva 'dir_dataset'


### VARIABILI
dir_documenti = 'documenti'  # Cartella documenti interi
dir_dataset = 'dataset/images' # Cartella definitva immagini parole tagliate (solo per numerazione)
dir_temp_images = 'dataset/temp_images' # Cartella dove vanno imm parole talgiate (per labeling)
numero_immagine = 1
numero_immagini_salvate = 0


### CONFIGURAZIONI
print("Configuro tools...")
detector = get_models.get_detector_model()


#### STAR PROCESS ####
print("\n\n####  START PROCESS ####")


# Prendi indice di numerazione più grande nel dataset
# (per cominciare a numerare da li  in poi)
larger_index_dataset = utile.get_larger_index_dataset(dir_dataset)
print('\nIndice più grande rilevato dataset:', larger_index_dataset, f'({dir_dataset})')

# Cicla su tutti i documenti
for file_name in os.listdir(dir_documenti) :
    if (file_name == '.DS_Store') :
            continue
    
    print(f'\n- Analizzo file: {file_name}...')
    
    doc_image = keras_ocr.tools.read(f'{dir_documenti}/{file_name}')
    
    print("Doc image read!")

    # Bounding-boxes immagine corrente
    boxes = detector.detect(images=[doc_image])[0]
    
    print("boxes fatte!")

    for idx,box in enumerate(boxes) :

        try :
            # Trasforma arrai in INTmo
            box = box.astype('int64')

            # Ritaglia bounding-box corrente da immagine...
            parola_cropped = doc_image[box[0][1]:box[2][1], box[0][0]:box[2][0] ]    

            # SALVA IMMAGINE IN CARTELLA -> 'image_dataset'
            nome_img_da_salvare = f'{dir_temp_images}/img_{larger_index_dataset + numero_immagine}.jpg'
            cv2.imwrite(nome_img_da_salvare, parola_cropped)
            print("- img salvata!")

            # incrementa contatore immagine
            numero_immagine = numero_immagine + 1

        except :
            print(f'- Errore in immagine: {file_name}')
            numero_immagini_salvate = numero_immagini_salvate - 1


    numero_immagini_salvate = numero_immagini_salvate + idx + 1
    print(f'- Bounding-boxes trovate doc corrente: {idx}')

print("\n\nFINE - Totale parole trovate (e salvate) :", numero_immagini_salvate)
    


