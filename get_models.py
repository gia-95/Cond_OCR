import keras_ocr
import tensorflow as tf
# tf.get_logger().setLevel('ERROR')



def get_recognition_trained_model() :
        
        alphabet = ' 0123456789ab./-()cdefghijklmnopqrstuvwxyz'
        recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))
        
        recognizer = keras_ocr.recognition.Recognizer(
              alphabet=recognizer_alphabet,
              weights='kurapan'
              )
        
        recognizer.model.load_weights('models/recogn_checkpoints_weights') 
        # recognizer.model = tf.keras.models.load_model('modelli_allenati/recogn_checkpoints_weights')
        recognizer.compile()

        return recognizer

def get_detector_model() :
       detector = keras_ocr.detection.Detector(weights='clovaai_general')
       return detector