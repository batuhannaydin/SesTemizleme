import wave
import numpy as np
import struct
import math 

def wav_dosyasi_okuma(path):
    try:
        with wave.open(path, 'rb') as ses_dosyasi:
            ornekleme_oran = ses_dosyasi.getframerate()
            ornekleme_sayisi = ses_dosyasi.getnframes()
            kanal_sayisi = ses_dosyasi.getnchannels()
            ham_ses_verisi = ses_dosyasi.readframes(ornekleme_sayisi)
            orneklemeler = np.frombuffer(ham_ses_verisi, dtype = np.int16).astype(np.float32) / 32768.0
                
            # stereo ses icin ilk kanali al
            if kanal_sayisi > 1:
                # sadece ilk kanal alinacak
                orneklemeler = orneklemeler[::kanal_sayisi] 
            return orneklemeler, ornekleme_oran
    # hata yakalama 
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} dosyasi bulunamadi.")
    except wave.Error as e:
        raise wave.Error(f"{path} dosyasi bulundu fakat okunurken hata meydana geldi: {e}")





