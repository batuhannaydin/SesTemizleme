import wave
import numpy as np
import struct
import math 
import matplotlib.pyplot as plt

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

def medyan_filtreleme(matrix, kernel_boyutu):
    satir, sutun = matrix.shape
    kernel_yukseklik, kernel_genislik = kernel_boyutu
    dolgu_yukseklik, dolgu_genislik = kernel_yukseklik // 2, kernel_genislik // 2
    dolgu_eklenmis_hali = np.pad(matrix, ((dolgu_yukseklik, dolgu_yukseklik), (dolgu_genislik, dolgu_genislik)), mode='edge')
    result = np.zeros_like(matrix)
    for i in range(satir):
        for j in range(sutun):
            pencere = dolgu_eklenmis_hali[i:i + kernel_yukseklik, j:j + kernel_genislik]
            result[i, j] = np.median(pencere)
    return result

def main():
    girdi_dosya = "kayit1.wav"
    cikti_dosya = "temizlenmis_kayit.wav"
    pencere_boyutu = 2048
    kaydirma_sayisi = 512
    kernel_boyutu = (1, 5)

    y, sr = wav_dosyasi_okuma(girdi_dosya)
    print(f"Orijinal (Temizlenmemiş / Ham sinyal uzunlugu: {len(y)} örnek ({len(y) / sr:.2f} saniye))")

    S_full = stft(y, pencere_boyutu, kaydirma_sayisi)
    faz = np.angle(S_full)
    genlik = np.abs(S_full)

    gurultu_gucu = np.mean(genlik[:, :int(sr * 0.1)], axis = 1)

    maske = genlik > gurultu_gucu[:, None]
    maske = maske.astype(float)
    maske = medyan_filtreleme(maske, kernel_boyutu = kernel_boyutu)

    S_clean = genlik * maske

    S_clean = S_clean * np.exp(1j * faz)

    y_clean = istft(S_clean, pencere_boyutu, kaydirma_sayisi)
    print(f"Temizlenmiş sinyal uzunluğu: {len(y_clean)} örnek ({len(y_clean) / sr:.2f} saniye)")
    
    sesi_kaydet(cikti_dosya, y_clean, sr)
    print(f"Temizlenmiş sinyal kaydedildi: '{cikti_dosya}'")
    print(f"Orijinal ve temizlenmiş sinyaller arasındaki fark: {len(y) - len(y_clean)} örnek")
    print("İşlem tamamlandı.")

    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(y, color = 'gray', label='Orijinal Sinyal')
    plt.title('Orijinal Sinyal')
    plt.xlabel('Örnek')
    plt.ylabel('Genlik')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(y_clean, color = 'blue', label='Temizlenmiş Sinyal')
    plt.title('Temizlenmiş Sinyal')
    plt.xlabel('Örnek')
    plt.ylabel('Genlik')
    plt.legend()

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()



