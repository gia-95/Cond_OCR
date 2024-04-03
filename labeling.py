import os
import cv2
import matplotlib.pyplot as plt
import get_models as models
import re
import utile
import shutil


### COMANDI
# '' -> (nessun inserimento) OK.
# 00 -> cancella KO.
# end -> chiudi processo

### VARIABILI 
dir_dataset_images = 'dataset/images' # Cartella immagini definitiva
dir_dataset_cf_images = 'dataset/images_cf' # Cartella immagini codici fiscali
dir_temp_images = 'dataset/temp_images' # Cartella temp immagini testi croppati
file_label_txt = 'labels/tuple.txt' # File con tuple totali
file_label_temp_txt = 'labels/tuple_temp.txt' # File con tuple temporaneo
file_label_CF_txt = 'labels/tuple_cf.txt' # File con tuple Codice Fiscale
train_labels = [] # Array di appoggio



### CONFIGURAZIONI
recognizer = models.get_recognition_trained_model()
fiscal_code_pattern = re.compile("^([a-z]{6}[0-9]{2}[a-z]{1}[0-9]{2}[a-z]{1}[0-9]{3}[a-z]{1})$|([0-9]{11})$")


### Per stamp es. 'Immagine 3 / 150'
indice_immagine_corrente = 1
n_immagini_tot = 0
for filename in os.listdir(dir_temp_images):
    n_immagini_tot = n_immagini_tot+1


# Apri i filetxt delle tuple
with open(file_label_txt, 'a') as f_definitivo, open(file_label_temp_txt, 'w') as f_temp: 

    # Cicla sulle immagini -> filename
    for filename in os.listdir(dir_temp_images):

        if (filename == '.DS_Store') :
            continue
        
        # Percorso immagine 
        pathFile = os.path.join(dir_temp_images, filename)

        # checking if it is a file
        if os.path.isfile(pathFile):

            print(f'\n- Immagine {indice_immagine_corrente} / {n_immagini_tot}  ({pathFile})')
            
            # Leggi immagine
            img = cv2.imread(pathFile)

            # Stampala
            fig, ax = plt.subplots()
            ax.imshow(img)
            plt.show(block=False)
            plt.pause(0.1)
            
            # Riconosci parola : recognition
            predicted_word =recognizer.recognize(img)
            print(f'  Predicted: {predicted_word}')

            # Inserisci parola stdout
            insered_word = input('  Inserisci: ')
            
            plt.close(fig)

            ###  Controlla inserimento
            match insered_word:

                case 'end': # Termina processo 
                    break
                
                case '00': # cancella elemeto (es. illegibile)
                    os.remove(pathFile)
                    print('  ...elemento eliminato!')
                    continue
                
                case '': # 'nessun inserimento' -> parola predicted OK
                    word_to_insert = predicted_word

                case _:
                    word_to_insert = insered_word
            

            ### ...inserisci tupla nei file txt
            tupla_to_insert = f'(\'{dir_dataset_images}/{filename}\', None, \'{word_to_insert}\'), '
            f_definitivo.write(tupla_to_insert)
            f_temp.write(tupla_to_insert)

            ### Sposta immagine in cartella finale
            shutil.copy(pathFile, dir_dataset_images)
            
            print("  Tupla scritta: ", tupla_to_insert)


            ### Check se è un codice  fiscale 
            ### (in caso inserisci in tuple_cf.txt, e copia imamgine nella cartella apposita)
            if (fiscal_code_pattern.match(word_to_insert)) :
                utile.inserisci_in_CF_file(word_to_insert, filename, file_label_CF_txt, dir_dataset_cf_images)
                shutil.copy(pathFile, dir_dataset_cf_images)

        
        indice_immagine_corrente = indice_immagine_corrente+1






