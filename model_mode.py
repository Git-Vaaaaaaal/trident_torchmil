from trident.slide_encoder_models import ABMILSlideEncoder, CHIEFSlideEncoder, FeatherSlideEncoder, GigaPathSlideEncoder, MadeleineSlideEncoder
from trident.slide_encoder_models import ThreadsSlideEncoder, TitanSlideEncoder, PRISMSlideEncoder, MeanSlideEncoder
from torchmil.models import ABMIL, CLAM_SB, DSMIL, TransMIL, DTFDMIL


def options(mode=int):
    if mode == 0 : #Threads
        PATCH_ENCODER = "conch_v15"
        encoder = ThreadsSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 1 : #Titan
        PATCH_ENCODER = "conch_v15"
        encoder = TitanSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 2 : #Prism
        PATCH_ENCODER = "virchow"
        encoder = PRISMSlideEncoder
        PATCH_SIZE = 224
        embedding_level = 2560
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 3 : #Chief
        PATCH_ENCODER = "ctranspath"
        encoder = CHIEFSlideEncoder
        PATCH_SIZE = 256
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 4 : #Gigapath
        PATCH_ENCODER = "virchow"
        encoder = GigaPathSlideEncoder
        PATCH_SIZE = 256
        embedding_level = 1536
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 5 : #Madeleine
        PATCH_ENCODER = "conch_v1"
        encoder = MadeleineSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 512
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 6 : #Prism
        PATCH_ENCODER = "conch_v15"
        encoder = FeatherSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    else :
        return print("Outvalue : between 0 to 6")
    

def options_torchmil(mode=int, marker=str, encoder=str):
    #Dictionnaire de taille de vecteur (embedding)
    dict_encoder = {
        "ThreadsSlideEncoder" : 768,
        "TitanSlideEncoder" : 768,
        "PRISMSlideEncoder" : 2560,
        "CHIEFSlideEncoder" : 768,
        "GigaPathSlideEncoder" : 1536,
        "MadeleineSlideEncoder" : 512,
        "FeatherSlideEncoder" : 768
    }
    if mode == 0 : #ABMIL
        encoder = encoder#s en branle
        coords_dir = f"extracted\\{encoder}\\coordinates\\{marker}"
        slide_labels_dir = f"extracted\\{encoder}\\slide_features_extraction\\{marker}"
        embedding_level = dict_encoder[encoder]
        model = ABMIL(in_dim=int(embedding_level), att_dim=128)
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 1 : #Titan
        PATCH_ENCODER = "conch_v15"
        encoder = TitanSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 2 : #Prism
        PATCH_ENCODER = "virchow"
        encoder = PRISMSlideEncoder
        PATCH_SIZE = 224
        embedding_level = 2560
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 3 : #Chief
        PATCH_ENCODER = "ctranspath"
        encoder = CHIEFSlideEncoder
        PATCH_SIZE = 256
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 4 : #Gigapath
        PATCH_ENCODER = "virchow"
        encoder = GigaPathSlideEncoder
        PATCH_SIZE = 256
        embedding_level = 1536
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 5 : #Madeleine
        PATCH_ENCODER = "conch_v1"
        encoder = MadeleineSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 512
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    elif mode == 6 : #Prism
        PATCH_ENCODER = "conch_v15"
        encoder = FeatherSlideEncoder
        PATCH_SIZE = 512
        embedding_level = 768
        return PATCH_ENCODER, encoder, PATCH_SIZE, embedding_level
    else :
        return print("Outvalue : between 0 to 6")