import keras_ocr
import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.experimental.output_all_intermediates(True)



def get_recognition_trained_model() :
        
        alphabet = ' 0123456789ab./-()cdefghijklmnopqrstuvwxyz'
        recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))
        
        recognizer = keras_ocr.recognition.Recognizer(
              alphabet=recognizer_alphabet,
              weights='kurapan'
              )
        
        # Questo non so perchè ma funziona, con warning, qualcosa carica.
        recognizer.model.load_weights('models/recogn_checkpoints_weights')
        
        # OK!
      #   recognizer.model  = tf.keras.models.load_model('models/my_model.h5') 

        # OK!
      #   recognizer.model = tf.keras.models.load_model('models/modello_test') 
        
        # NON FUNZIONA.... nonostante la cartella sia uguale a quell'altra, lo devi da come pesi!
      #   recognizer.model = tf.keras.models.load_model('models/recogn_checkpoints_weights') 
        
        recognizer.compile()

        return recognizer

def get_detector_model() :
       detector = keras_ocr.detection.Detector(weights='clovaai_general')
       return detector