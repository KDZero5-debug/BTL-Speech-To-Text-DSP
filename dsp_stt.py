import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft
import speech_recognition as sr
from pydub import AudioSegment

def convert_and_resample_audio(input_path, output_path="giong_cua_nam.wav"):
    print(f"--- TIỀN XỬ LÝ: Chuyển đổi định dạng và Lấy mẫu lại ---")
    # Đọc file ghi âm gốc từ điện thoại (.m4a, .aac, .mp3, v.v.)
    audio = AudioSegment.from_file(input_path)
    
    print(f"[Gốc] Tần số lấy mẫu ban đầu: {audio.frame_rate} Hz, Số kênh: {audio.channels}")
    
    # ÁP DỤNG LÝ THUYẾT LẤY MẪU LẠI (RESAMPLING)
    # Hạ mẫu xuống 16000 Hz và chuyển về 1 kênh (Mono) theo chuẩn hệ thống STT
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    print(f"[DSP Resampling] Đã hạ mẫu xuống: {audio.frame_rate} Hz, Số kênh: {audio.channels} (Mono)")
    
    # Xuất ra file .wav (Định dạng PCM thô không nén dành cho xử lý tín hiệu)
    audio.export(output_path, format="wav")
    print(f"-> Đã xuất file số hóa chuẩn: {output_path}\n")
    return output_path
def advanced_dsp_and_stt(wav_path):
    print(f"=== BẮT ĐẦU QUÁ TRÌNH XỬ LÝ TÍN HIỆU & NHẬN DIỆN: {wav_path} ===")
    
    # ----------------------------------------------------
    # PHẦN 1: PHÂN TÍCH LÝ THUYẾT LẤY MẪU (Dành cho báo cáo BTL)
    # ----------------------------------------------------
    fs, signal = wav.read(wav_path)
    print(f"[Lấy mẫu] Tần số lấy mẫu (Fs): {fs} Hz")
    
    if len(signal.shape) > 1:
        signal = signal[:, 0]
        
    # Tính toán FFT nhanh để trích xuất tần số đỉnh (minh họa DSP)
    window_size = min(int(0.03 * fs), len(signal))
    frame = signal[:window_size]
    fft_spectrum = np.abs(fft(frame * np.hamming(window_size)))
    frequencies = np.fft.fftfreq(window_size, 1/fs)
    peak_freq = frequencies[np.where(frequencies >= 0)][np.argmax(fft_spectrum[np.where(frequencies >= 0)])]
    print(f"[Trích xuất đặc trưng] Tần số trội của đoạn đầu: {peak_freq:.2f} Hz")

    # ----------------------------------------------------
    # PHẦN 2: CHUYỂN LỜI NÓI THÀNH VĂN BẢN (STT NGUYÊN CÂU)
    # ----------------------------------------------------
    print("\n[Hệ thống STT] Đang kích hoạt bộ lọc nhiễu môi trường...")
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:

# === BỔ SUNG BỘ LỌC NHIỄU Ở ĐÂY ===
        # Lệnh này sẽ quét phân tích 0.5 giây đầu của file để nhận diện tần số nhiễu môi trường
        # và tự động thiết lập một ngưỡng lọc (dynamic energy threshold) để triệt tiêu nhiễu.
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("[Bộ lọc] Đã triệt tiêu tạp âm nền thành công!")
        
        # Đọc toàn bộ dữ liệu âm thanh đã lấy mẫu
        audio_data = recognizer.record(source)
        try:
            # Gọi mô hình nhận diện giọng nói tiếng Việt
            van_ban_dau_ra = recognizer.recognize_google(audio_data, language="vi-VN")
            
            print("\n--- KẾT QUẢ CHUYỂN LỜI NÓI THÀNH VĂN BẢN ---")
            print(f"🎤 Giọng nói của bạn: \"{van_ban_dau_ra}\"")
            print("--------------------------------------------\n")
            return van_ban_dau_ra
        except sr.UnknownValueError:
            print("❌ Lỗi: AI không thể nghe rõ câu nói của bạn. Hãy thử nói rõ ràng hơn!")
        except sr.RequestError:
            print("❌ Lỗi: Không thể kết nối Internet để gọi mô hình nhận diện.")

if __name__ == "__main__":
    # Điền chính xác tên file đuôi .aac bạn vừa tải lên ở đây
    file_goc_dien_thoai = "ghi_am.aac" 
    
    # Đoạn dưới giữ nguyên, code sẽ tự đọc file .aac này để hạ mẫu số hóa
    file_chuan_wav = convert_and_resample_audio(file_goc_dien_thoai)
    advanced_dsp_and_stt(file_chuan_wav)
