from keras.models import load_model

def get_model():
    path_to_model = 'C:/Users/jkee2/Desktop/volnolomi/MODELS/AE_like'
    my_model = load_model(path_to_model)

    return my_model