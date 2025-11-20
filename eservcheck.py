import pandas as pd

# --- 1. Definisikan Kata Kunci Prioritas Tinggi ---
# Prioritas TERTINGGI: Masalah Keandalan (Reliability - Hasil Negatif)
KW_RELIABILITY = ['gagal', 'refund', 'tipu', 'penipu', 'hilang', 'dana', 'kurir', 'pengirim', 
                  'tertunda', 'salah kirim', 'blokir', 'ditolak', 'dicancel', 'uang']

# Prioritas KEDUA: Masalah Responsiveness (Layanan Manusia/Komplain)
KW_RESPONSIVENESS = ['cs', 'respon', 'komplain', 'dijawab', 'lambat', 'balas', 'template']

# Prioritas KETIGA: Masalah Kinerja/Proses (Ease of Use)
KW_EASE_OF_USE = ['lemot', 'bug', 'error', 'nge-lag', 'loading', 'ribet', 'susah', 'crash', 'ngeblank']

# Prioritas KEEMPAT: Masalah Harga/Promo (Information Quality)
KW_INFO_QUALITY = ['promo', 'voucher', 'ongkir', 'mahal', 'harga', 'skon', 'diskon']


# --- 2. Fungsi Koreksi Utama ---
def apply_consistency_correction(row):
    # Pastikan ulasan_text diambil dari kolom yang benar ('Ulasan' setelah di-rename)
    ulasan_text = str(row['Ulasan']).lower()
    original_label = row['Original_Label']
    
    # C. Pengecualian: Jika label saat ini sudah 'privacy' atau 'webdesign', pertahankan.
    if original_label in ['privacy', 'webdesign']:
        return original_label

    # A. Cek Prioritas TERTINGGI: Reliability
    if any(kw in ulasan_text for kw in KW_RELIABILITY):
        if original_label in ['easeofuse', 'responsiveness', 'nonquality']:
            return 'reliability'
    
    # B. Cek Prioritas KEDUA: Responsiveness
    if any(kw in ulasan_text for kw in KW_RESPONSIVENESS):
        if original_label in ['easeofuse', 'nonquality']:
            return 'responsiveness'

    # D. Cek Prioritas KEEMPAT (Info Quality - untuk ulasan umum/Nonquality)
    if original_label == 'nonquality' and any(kw in ulasan_text for kw in KW_INFO_QUALITY):
        return 'informationquality'

    # E. Cek Prioritas KETIGA (Ease of Use - untuk ulasan umum/Nonquality)
    if original_label == 'nonquality' and any(kw in ulasan_text for kw in KW_EASE_OF_USE):
        return 'easeofuse'

    # Jika tidak ada prioritas yang jelas atau label aslinya sudah benar, pertahankan label asli.
    return original_label

# --- 3. Proses Muat Data dan Koreksi ---
try:
    df = pd.read_csv("5000_cleaned_combined2.csv")
    
    # LANGKAH PERBAIKAN KEYERROR: Mengganti nama kolom yang benar
    # Mengganti nama kolom label 'label' menjadi 'Original_Label'
    # Mengganti nama kolom ulasan 'cleaned_content' menjadi 'Ulasan'
    df = df.rename(columns={'label': 'Original_Label', 'cleaned_content': 'Ulasan'}) 
    
    # Membuat kolom 'Label' baru yang akan diisi hasil koreksi
    df['Label'] = df['Original_Label'] 

except FileNotFoundError:
    print("Error: File '5000_cleaned_combined2.csv' tidak ditemukan. Pastikan nama file dan direktori sudah benar.")
    exit()

# Terapkan fungsi koreksi ke seluruh Dataset
df['Label'] = df.apply(apply_consistency_correction, axis=1)

# --- 4. Simpan File Baru yang Sudah Dikoreksi ---
output_filename = "5000_cleaned_combined_KOREKSI_SVM.csv"

# Hanya simpan kolom 'Ulasan' (teks asli) dan 'Label' (label terkoreksi)
df[['Ulasan', 'Label']].to_csv(output_filename, index=False)

print(f"\n✅ Koreksi konsistensi label selesai.")
print(f"File baru '{output_filename}' telah dibuat. Silakan gunakan file ini untuk melatih model SVM Anda.")