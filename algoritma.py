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

def sesi_kaydet(path, orneklemeler, ornekleme_orani):
    try:
        orneklemeler_int16 = (orneklemeler * 32767).astype(np.int16)
        with wave.open(path, 'wb') as ses_dosyasi:
            ses_dosyasi.setnchannels(1)
            ses_dosyasi.setsampwidth(2)
            ses_dosyasi.setframerate(ornekleme_orani)
            ses_dosyasi.writeframes(orneklemeler_int16.tobytes())
    except Exception as e:
        raise Exception(f"{path} dosyasina yazarken hata meydana geldi: {e}")

def stft(ses_sinyali, pencere_boyutu = 2048, kaydirma_sayisi = 512):
    pencere_boyutu = min(pencere_boyutu, len(ses_sinyali))
    pencere = np.hanning(pencere_boyutu)
    cerceve_sayisi = 1 + int((len(ses_sinyali) - pencere_boyutu) / kaydirma_sayisi)
    cerceve_sayisi = max(1,cerceve_sayisi)
    stft_matrix = np.zeros((pencere_boyutu // 2 + 1, cerceve_sayisi), dtype=np.complex64)
    for i in range(cerceve_sayisi):
        baslangic = i * kaydirma_sayisi
        bitis = min(baslangic + pencere_boyutu, len(ses_sinyali))
        if bitis - baslangic < pencere_boyutu:
            cerceve = np.zeros(pencere_boyutu)
            cerceve[:bitis - baslangic] = ses_sinyali[baslangic:bitis]
        else:
            cerceve = ses_sinyali[baslangic:bitis]
            cerceve = cerceve * pencere
            spektrum = np.fft.rfft(cerceve)
            stft_matrix[:, i] = spektrum
    return stft_matrix

def istft(stft_matrix, pencere_boyutu = 2048, kaydirma_sayisi = 512):
    pencere = np.hanning(pencere_boyutu)
    cerceve_sayisi = stft_matrix.shape[1]
    ses_sinyali_length = (cerceve_sayisi - 1) * kaydirma_sayisi + pencere_boyutu
    ses_sinyali = np.zeros(ses_sinyali_length)
    for i in range(cerceve_sayisi):
        baslangic = i * kaydirma_sayisi
        bitis = baslangic + pencere_boyutu
        cerceve = np.fft.irfft(stft_matrix[:, i])
        if len(cerceve) < pencere_boyutu:
            temp = np.zeros(pencere_boyutu)
            temp[:len(cerceve)] = cerceve
            cerceve = temp
        ses_sinyali[baslangic:bitis] += cerceve * pencere[:len(cerceve)]
    return ses_sinyali






